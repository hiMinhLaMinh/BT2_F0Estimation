"""
Lõi thuật toán ước lượng F0: framing, Normalized ACF, AMDF, AMDF+YIN.

Phương pháp khuyến nghị chính: Normalized ACF (Autocorrelation Function) với:
- Chuẩn hóa Pearson không thiên lệch theo độ trễ (lag).
- Cơ chế chọn đỉnh cực đại nổi bật đầu tiên (first prominent peak) chống nhảy quãng tám (octave halving).
- Nội suy parabol quanh đỉnh để đạt độ chính xác dưới mẫu (sub-sample).
- Bộ lọc năng lượng (Energy Gating) và lọc trung vị 5 điểm (5-point Median Filter) khử điểm ngoại lai.

Vẫn giữ nguyên các hàm AMDF và AMDF+YIN trước đó để đối chiếu.
Chỉ dùng numpy và scipy theo ràng buộc của bài tập.
"""

import numpy as np

FRAME_LENGTH_MS = 25  # Cố định để xếp hạng
FRAME_SHIFT_MS = 10


def frame_signal(signal, fs, frame_length_ms=FRAME_LENGTH_MS, frame_shift_ms=FRAME_SHIFT_MS):
    """Chia tín hiệu thành các khung. Trả về (frames, centers_sec)."""
    frame_len = int(round(fs * frame_length_ms / 1000))
    frame_shift = int(round(fs * frame_shift_ms / 1000))

    starts = list(range(0, len(signal) - frame_len + 1, frame_shift))
    frames = [signal[s : s + frame_len] for s in starts]
    centers_sec = [(s + frame_len / 2) / fs for s in starts]
    return frames, centers_sec


def f0_range_to_lag_range(fs, f0_min=70, f0_max=400):
    """Đổi dải F0 (Hz) sang dải lag (mẫu): f0_max -> lag_min, f0_min -> lag_max."""
    lag_min = int(np.floor(fs / f0_max))
    lag_max = int(np.ceil(fs / f0_min))
    return lag_min, lag_max


def short_time_acf(frame, lag_min, lag_max):
    """
    ACF ngắn hạn cơ bản R(k) = sum x(n) x(n+k).
    Returns: lags, acf (thô), acf_norm (= R(k)/R(0))
    """
    frame = np.asarray(frame, dtype=np.float64)
    N = len(frame)
    lag_max = min(lag_max, N - 1)

    lags = np.arange(lag_min, lag_max + 1)
    acf = np.empty(len(lags), dtype=np.float64)
    for i, k in enumerate(lags):
        acf[i] = np.dot(frame[: N - k], frame[k:N])

    r0 = np.dot(frame, frame)
    acf_norm = acf / r0 if r0 > 0 else np.zeros_like(acf)
    return lags, acf, acf_norm


def short_time_acf_normalized(frame, lag_min, lag_max, threshold=0.0, rel_peak_ratio=0.80):
    """
    Hàm tự tương quan chuẩn hóa không thiên lệch (Normalized Cross-Correlation / Pearson ACF).
    R_norm(k) = sum(x[n] * x[n+k]) / sqrt(sum(x[n]^2) * sum(x[n+k]^2))
    
    Giá trị nằm trong [-1, 1]. Khung hữu thanh tuần hoàn đạt đỉnh gần 1.0.
    Áp dụng thuật toán chọn đỉnh địa phương nổi bật đầu tiên (first prominent peak)
    để ngăn ngừa lỗi chia đôi tần số (pitch halving / bắt nhầm 2*T0).
    
    Returns:
        lags (np.ndarray): mảng các độ trễ lag
        acfs (np.ndarray): mảng giá trị normalized ACF
        chosen_lag (float): độ trễ tối ưu sau khi nội suy parabol
        confidence (float): biên độ đỉnh ACF (dùng để phân ngưỡng V/UV)
        energy (float): năng lượng phương sai của khung (STE)
    """
    frame = np.asarray(frame, dtype=np.float64)
    N = len(frame)
    lag_max = min(lag_max, N - 2)
    lags = np.arange(lag_min, lag_max + 1)

    fr_c = frame - np.mean(frame)
    energy = float(np.mean(fr_c ** 2))

    acfs = np.zeros(len(lags), dtype=np.float64)
    for i, k in enumerate(lags):
        x1 = fr_c[: N - k]
        x2 = fr_c[k:N]
        denom = np.sqrt(np.dot(x1, x1) * np.dot(x2, x2))
        acfs[i] = np.dot(x1, x2) / denom if denom > 1e-12 else 0.0

    # Tìm các đỉnh cực đại địa phương (local peaks)
    peaks = []
    for i in range(1, len(acfs) - 1):
        if acfs[i] > acfs[i - 1] and acfs[i] > acfs[i + 1] and acfs[i] > 0:
            peaks.append(i)

    if not peaks:
        global_max = float(np.max(acfs)) if len(acfs) else 0.0
        best_idx = int(np.argmax(acfs)) if len(acfs) else 0
        return lags, acfs, float(lags[best_idx]), global_max, energy

    global_max = float(np.max([acfs[p] for p in peaks]))

    # Chọn đỉnh nổi bật đầu tiên có chiều cao >= rel_peak_ratio * global_max
    chosen_idx = peaks[int(np.argmax([acfs[p] for p in peaks]))]
    for p in peaks:
        if acfs[p] >= rel_peak_ratio * global_max and acfs[p] >= threshold:
            chosen_idx = p
            break

    # Nội suy parabol 3 điểm quanh đỉnh để đạt độ chính xác dưới mẫu (sub-sample)
    lag_refined = float(lags[chosen_idx])
    if 0 < chosen_idx < len(acfs) - 1:
        a, b, c = acfs[chosen_idx - 1], acfs[chosen_idx], acfs[chosen_idx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            lag_refined += 0.5 * (a - c) / denom

    confidence = float(acfs[chosen_idx])
    return lags, acfs, lag_refined, confidence, energy


def short_time_amdf(frame, lag_min, lag_max):
    """
    AMDF ngắn hạn: AMDF(k) = mean |x(n) - x(n+k)| trên các cặp mẫu chồng lấp.
    Returns: lags, amdf
    """
    frame = np.asarray(frame, dtype=np.float64)
    N = len(frame)
    lag_max = min(lag_max, N - 1)

    lags = np.arange(lag_min, lag_max + 1)
    amdf = np.empty(len(lags), dtype=np.float64)
    for i, k in enumerate(lags):
        amdf[i] = np.mean(np.abs(frame[: N - k] - frame[k:N]))
    return lags, amdf


def short_time_amdf_yin(frame, lag_min, lag_max, threshold=0.1):
    """AMDF kết hợp YIN (giữ lại theo mã nguồn gốc)."""
    frame = np.asarray(frame, dtype=np.float64)
    frame = frame - frame.mean()
    N = len(frame)
    lag_max = min(lag_max, N - 2)
    W = N - lag_max

    d = np.zeros(lag_max + 1)
    ref = frame[:W]
    for k in range(1, lag_max + 1):
        d[k] = np.mean(np.abs(ref - frame[k : k + W]))

    d_norm = np.ones(lag_max + 1)
    running = np.cumsum(d)
    k_arr = np.arange(1, lag_max + 1)
    d_norm[1:] = d[1:] / (running[1:] / k_arr)

    lags = np.arange(lag_min, lag_max + 1)

    chosen = None
    for k in lags:
        if d_norm[k] < threshold:
            while k + 1 <= lag_max and d_norm[k + 1] < d_norm[k]:
                k += 1
            chosen = k
            break
    if chosen is None:
        chosen = lags[np.argmin(d_norm[lags])]

    # Nội suy parabol quanh dip
    lag_refined = float(chosen)
    if 1 <= chosen < lag_max:
        a, b, c = d_norm[chosen - 1], d_norm[chosen], d_norm[chosen + 1]
        denom = a - 2 * b + c
        if denom > 1e-12:
            lag_refined = chosen + 0.5 * (a - c) / denom

    confidence = 1.0 - d_norm[chosen]
    return lags, d, d_norm, lag_refined, confidence


def estimate_f0_track(
    signal,
    fs,
    f0_min=70,
    f0_max=400,
    vuv_threshold=0.59,
    yin_threshold=0.1,
    frame_length_ms=FRAME_LENGTH_MS,
    frame_shift_ms=FRAME_SHIFT_MS,
    method="acf",
    apply_median_filter=True,
    median_kernel_size=5,
):
    """
    Tính toán mảng F0 theo thời gian cho toàn bộ tín hiệu.

    Phương pháp:
        - 'acf' (Khuyến nghị): Dùng Normalized ACF + Energy Gating.
        - 'amdf_yin': Dùng AMDF-YIN nguyên bản.

    Tham số lọc:
        - apply_median_filter (bool): Bật/tắt bộ lọc trung vị làm mượt F0.
        - median_kernel_size (int): Kích thước kernel lọc trung vị (mặc định: 5 điểm).

    Returns:
        f0_track (np.ndarray): Mảng chứa giá trị F0 (Hz) tương ứng với mỗi khung (Unvoiced = np.nan).
        centers_sec (list): Mảng thời điểm tâm (giây) của mỗi khung.
    """
    frames, centers_sec = frame_signal(signal, fs, frame_length_ms, frame_shift_ms)
    lag_min, lag_max = f0_range_to_lag_range(fs, f0_min, f0_max)

    f0_track = []

    if method == "acf":
        # Tính năng lượng ngắn hạn STE cho từng khung
        energies = np.array([float(np.mean((fr - np.mean(fr)) ** 2)) for fr in frames])
        # Ngưỡng khoảng lặng (loại bỏ các khung không có tiếng nói)
        sil_thresh = max(1e-5, float(np.max(energies)) * 0.005)

        for fr, e in zip(frames, energies):
            if e < sil_thresh:
                f0_track.append(np.nan)
                continue

            lags, acfs, chosen_lag, confidence, _ = short_time_acf_normalized(
                fr, lag_min, lag_max, threshold=vuv_threshold
            )

            if confidence >= vuv_threshold and chosen_lag > 0:
                f0 = fs / chosen_lag
                f0_track.append(f0)
            else:
                f0_track.append(np.nan)

    elif method == "amdf_yin":
        for frame in frames:
            _, _, _, chosen_lag, confidence = short_time_amdf_yin(
                frame, lag_min, lag_max, threshold=yin_threshold
            )
            if confidence >= vuv_threshold and chosen_lag > 0:
                f0 = fs / chosen_lag
                f0_track.append(f0)
            else:
                f0_track.append(np.nan)
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}. Chọn 'acf' hoặc 'amdf_yin'.")

    f0_arr = np.array(f0_track, dtype=np.float64)

    # Bộ lọc trung vị (5-point median filter) trên các phân đoạn voiced liên tiếp
    if apply_median_filter and len(f0_arr) >= 3:
        smoothed = f0_arr.copy()
        is_voiced = ~np.isnan(f0_arr)

        # Nhóm các chỉ số khung voiced liên tiếp thành các khối (blocks)
        blocks = []
        curr = []
        for i, v in enumerate(is_voiced):
            if v:
                curr.append(i)
            elif curr:
                blocks.append(curr)
                curr = []
        if curr:
            blocks.append(curr)

        half_w = median_kernel_size // 2

        for block in blocks:
            blen = len(block)
            if blen < 3:
                continue
            for pos in range(blen):
                # Bán kính thích ứng tại biên phân đoạn
                w = min(pos, blen - 1 - pos, half_w)
                if w >= 1:
                    window = [f0_arr[block[pos + d]] for d in range(-w, w + 1)]
                    smoothed[block[pos]] = float(np.median(window))
        f0_arr = smoothed

    return f0_arr, centers_sec
