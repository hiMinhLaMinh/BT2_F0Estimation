"""
Lõi thuật toán ước lượng F0: framing, ACF, AMDF, AMDF+YIN.

ACF và AMDF là bản baseline (định nghĩa giáo trình, chưa cải tiến).
Chỉ dùng numpy theo ràng buộc của bài tập.
"""

import numpy as np

FRAME_LENGTH_MS = 25  # cố định để xếp hạng
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
    ACF ngắn hạn R(k) = sum x(n) x(n+k).

    Returns: lags, acf (thô), acf_norm (= R(k)/R(0), trong [-1, 1])
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
    """
    AMDF + cumulative mean normalization của YIN (de Cheveigne & Kawahara, 2002),
    giữ hiệu số |.| để nhất quán với AMDF.

    - Chuẩn hóa theo trung bình tích lũy: khử falling trend.
    - Chọn lag bằng "dip đầu tiên dưới ngưỡng": tránh octave error.

    Returns: lags, d, d_norm, chosen_lag, confidence
        d, d_norm : k = 0..lag_max (cần đủ từ k=0 để trung bình tích lũy đúng)
        confidence: 1 - d_norm[chosen_lag]; cao = tuần hoàn mạnh (voiced)
    """
    frame = np.asarray(frame, dtype=np.float64)
    N = len(frame)
    lag_max = min(lag_max, N - 1)

    d = np.zeros(lag_max + 1, dtype=np.float64)
    for k in range(1, lag_max + 1):
        d[k] = np.sum(np.abs(frame[: N - k] - frame[k:N]))

    d_norm = np.ones(lag_max + 1, dtype=np.float64)  # d'(0) = 1 theo quy ước YIN
    running_sum = 0.0
    for k in range(1, lag_max + 1):
        running_sum += d[k]
        d_norm[k] = d[k] / (running_sum / k)

    lags = np.arange(lag_min, lag_max + 1)

    chosen_lag = None
    for k in lags:
        if d_norm[k] < threshold:
            while k + 1 <= lag_max and d_norm[k + 1] < d_norm[k]:
                k += 1
            chosen_lag = k
            break

    if chosen_lag is None:  # không dip nào qua ngưỡng -> khung nhiều khả năng unvoiced
        chosen_lag = lags[np.argmin(d_norm[lags])]

    confidence = 1.0 - d_norm[chosen_lag]
    return lags, d, d_norm, chosen_lag, confidence
