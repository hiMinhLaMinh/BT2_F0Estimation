"""
visualization.py - Vẽ đồ thị Waveform, F0 Contour và Minh họa 2 khung ACF theo chuẩn thẩm mỹ cao.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from io_utils import load_wav_normalized, parse_lab_file, label_at_time
from pitch_estimation import (
    frame_signal,
    f0_range_to_lag_range,
    short_time_acf_normalized,
    FRAME_LENGTH_MS,
    FRAME_SHIFT_MS,
)


def plot_f0_contour(wav_path, lab_path, f0_track, centers_sec, output_path, method_name="Normalized ACF"):
    """
    Vẽ 2 subplot:
    1. Waveform tín hiệu gốc.
    2. F0 Contour (đường F0 ước lượng + các mảng Voiced Ground Truth từ file .lab).
    """
    fs, signal = load_wav_normalized(wav_path)
    time_axis = np.arange(len(signal)) / fs

    segments, f0mean_gt, f0std_gt = parse_lab_file(lab_path)

    # Tính F0mean ước lượng để hiển thị trên đồ thị
    valid_f0 = f0_track[np.isfinite(f0_track) & (f0_track > 0)]
    f0mean_est = float(np.mean(valid_f0)) if len(valid_f0) else float("nan")
    f0std_est = float(np.std(valid_f0)) if len(valid_f0) else float("nan")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Subplot 1: Waveform tín hiệu gốc
    ax1.plot(time_axis, signal, color="#1e293b", linewidth=0.6)
    title_file = Path(wav_path).name
    ax1.set_title(f"Waveform: {title_file} (fs = {fs} Hz)", fontsize=13, fontweight="bold", pad=8)
    ax1.set_ylabel("Biên độ", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: F0 Contour
    # Vẽ các vùng Voiced theo Ground Truth (nền xanh lá nhạt)
    has_gt_legend = False
    for start, end, label in segments:
        if label == "v":
            lbl = "Ground Truth Voiced (.lab)" if not has_gt_legend else None
            ax2.axvspan(start, end, color="#22c55e", alpha=0.22, label=lbl)
            has_gt_legend = True

    # Vẽ đường F0 ước lượng
    ax2.plot(
        centers_sec,
        f0_track,
        color="#2563eb",
        marker=".",
        linestyle="none",
        markersize=5,
        label=f"F0 Ước lượng ({method_name})",
    )

    # Đường tham chiếu F0mean chuẩn (nếu có)
    if f0mean_gt is not None:
        ax2.axhline(
            y=f0mean_gt,
            color="#dc2626",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label=f"F0mean chuẩn = {f0mean_gt:.1f} Hz (est = {f0mean_est:.1f} Hz)",
        )

    ax2.set_title(
        f"Pitch Contour (F0) - Ước lượng: {f0mean_est:.1f} ± {f0std_est:.1f} Hz "
        f"| Chuẩn: {f0mean_gt:.1f} ± {f0std_gt:.1f} Hz",
        fontsize=12,
        fontweight="semibold",
        pad=8,
    )
    ax2.set_ylabel("Tần số (Hz)", fontsize=11)
    ax2.set_xlabel("Thời gian (giây)", fontsize=11)
    ax2.set_ylim([50, 450])  # Dải F0 khảo sát
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Tối ưu legend
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right", framealpha=0.9)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_two_frames_acf(
    wav_path,
    t_voiced=None,
    t_unvoiced=None,
    frame_idx_v=None,
    frame_idx_u=None,
    output_path=None,
    threshold=0.591812,
    f0_min=70,
    f0_max=400,
    frame_length_ms=FRAME_LENGTH_MS,
    frame_shift_ms=FRAME_SHIFT_MS,
):
    """
    Minh họa trực quan 2 khung tín hiệu theo yêu cầu mục 4 của đề bài:
    - 1 khung chứa tiếng nói tuần hoàn (hữu thanh - Voiced) -> ACF có cực đại rõ rệt tại T0 -> tính được F0.
    - 1 khung chứa tiếng nói không tuần hoàn (vô thanh - Unvoiced) -> ACF không có cực đại vượt ngưỡng -> F0 không xác định.

    Người dùng có thể tự chọn khung qua:
    - Thời điểm (giây): t_voiced, t_unvoiced
    - Hoặc chỉ số khung: frame_idx_v, frame_idx_u
    - Hoặc để trống: hàm sẽ tự động chọn 1 khung Voiced và 1 khung Unvoiced tiêu biểu nhất.

    Returns:
        info_dict: dict chứa thông số chi tiết (t, F0, T0, peak confidence) của 2 khung.
    """
    wav_path = Path(wav_path)
    fs, signal = load_wav_normalized(wav_path)
    frames, centers_sec = frame_signal(signal, fs, frame_length_ms, frame_shift_ms)
    lag_min, lag_max = f0_range_to_lag_range(fs, f0_min, f0_max)
    total_frames = len(frames)

    # Đọc nhãn nếu có file .lab đi kèm
    lab_path = wav_path.with_suffix(".lab")
    segments = parse_lab_file(lab_path)[0] if lab_path.exists() else []

    # 1. Xác định khung Hữu thanh (Voiced)
    if frame_idx_v is not None:
        idx_v = max(0, min(total_frames - 1, int(frame_idx_v)))
    elif t_voiced is not None:
        idx_v = int(np.argmin([abs(c - t_voiced) for c in centers_sec]))
    else:
        # Tự động tìm khung Voiced có peak ACF cao nhất và nhãn 'v'
        best_v_idx = 0
        best_v_conf = -1.0
        for i, (fr, t_c) in enumerate(zip(frames, centers_sec)):
            lbl = label_at_time(t_c, segments) if segments else "v"
            if lbl == "v" or not segments:
                _, _, _, conf, _ = short_time_acf_normalized(fr, lag_min, lag_max, threshold=0.0)
                if conf > best_v_conf:
                    best_v_conf = conf
                    best_v_idx = i
        idx_v = best_v_idx

    # 2. Xác định khung Vô thanh (Unvoiced)
    if frame_idx_u is not None:
        idx_u = max(0, min(total_frames - 1, int(frame_idx_u)))
    elif t_unvoiced is not None:
        idx_u = int(np.argmin([abs(c - t_unvoiced) for c in centers_sec]))
    else:
        # Tự động tìm khung Unvoiced tiêu biểu (nhãn 'uv' hoặc năng lượng vừa phải nhưng peak thấp)
        best_u_idx = 0
        best_u_score = 999.0
        for i, (fr, t_c) in enumerate(zip(frames, centers_sec)):
            lbl = label_at_time(t_c, segments) if segments else "uv"
            if lbl == "uv" or not segments:
                _, _, _, conf, eng = short_time_acf_normalized(fr, lag_min, lag_max, threshold=0.0)
                if eng > 1e-5 and conf < best_u_score:
                    best_u_score = conf
                    best_u_idx = i
        idx_u = best_u_idx

    # Tính toán cho khung Voiced
    v_frame = frames[idx_v]
    v_t_center = centers_sec[idx_v]
    v_time_ms = (np.arange(len(v_frame))) / fs * 1000
    v_lags, v_acfs, v_lag_opt, v_conf, v_energy = short_time_acf_normalized(
        v_frame, lag_min, lag_max, threshold=threshold
    )
    v_t0_ms = v_lag_opt / fs * 1000
    v_f0 = fs / v_lag_opt if v_lag_opt > 0 else float("nan")

    # Tính toán cho khung Unvoiced
    u_frame = frames[idx_u]
    u_t_center = centers_sec[idx_u]
    u_time_ms = (np.arange(len(u_frame))) / fs * 1000
    u_lags, u_acfs, u_lag_opt, u_conf, u_energy = short_time_acf_normalized(
        u_frame, lag_min, lag_max, threshold=threshold
    )
    u_t0_ms = u_lag_opt / fs * 1000
    u_is_voiced = (u_conf >= threshold)
    u_f0 = (fs / u_lag_opt) if u_is_voiced else float("nan")

    # --- KHỞI TẠO ĐỒ THỊ 3 TẦNG (PREMIUM AESTHETICS) ---
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.25, 1.25], hspace=0.42, wspace=0.28)

    # 1. HÀNG 1: Dạng sóng toàn bộ tín hiệu (chỉ rõ vị trí trích xuất của 2 khung)
    ax_full = fig.add_subplot(gs[0, :])
    full_time = np.arange(len(signal)) / fs
    ax_full.plot(full_time, signal, color="#334155", linewidth=0.5, label="Toàn bộ tín hiệu")
    
    # Đánh dấu vị trí khung Voiced và Unvoiced trên toàn bộ file
    fl_sec = frame_length_ms / 1000.0
    v_start = max(0.0, v_t_center - fl_sec / 2)
    v_end = min(full_time[-1], v_t_center + fl_sec / 2)
    u_start = max(0.0, u_t_center - fl_sec / 2)
    u_end = min(full_time[-1], u_t_center + fl_sec / 2)

    ax_full.axvspan(v_start, v_end, color="#2563eb", alpha=0.35, label=f"Khung Voiced (t={v_t_center:.3f}s)")
    ax_full.axvspan(u_start, u_end, color="#dc2626", alpha=0.35, label=f"Khung Unvoiced (t={u_t_center:.3f}s)")
    ax_full.axvline(x=v_t_center, color="#1d4ed8", linestyle="--", linewidth=1.5)
    ax_full.axvline(x=u_t_center, color="#b91c1c", linestyle="--", linewidth=1.5)
    ax_full.set_title(
        f"Vị trí 2 khung tín hiệu được chọn trên toàn bộ tín hiệu: {wav_path.name} (fs = {fs} Hz)",
        fontsize=12,
        fontweight="bold",
        pad=6,
    )
    ax_full.set_xlabel("Thời gian (giây)", fontsize=10)
    ax_full.set_ylabel("Biên độ", fontsize=10)
    ax_full.grid(True, linestyle="--", alpha=0.5)
    ax_full.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # 2. HÀNG 2 - CỘT 1: Waveform Khung Hữu Thanh (Voiced)
    ax_v_wave = fig.add_subplot(gs[1, 0])
    ax_v_wave.plot(v_time_ms, v_frame, color="#1d4ed8", linewidth=1.2)
    ax_v_wave.set_title(
        f"Khung Hữu Thanh (Voiced) [Index={idx_v}, t={v_t_center:.3f}s]\n"
        f"Dao động tuần hoàn điều hòa, T0 ≈ {v_t0_ms:.2f} ms",
        fontsize=10.5,
        fontweight="bold",
        color="#1e3a8a",
    )
    ax_v_wave.set_xlabel("Thời gian trong khung (ms)", fontsize=10)
    ax_v_wave.set_ylabel("Biên độ", fontsize=10)
    ax_v_wave.grid(True, linestyle="--", alpha=0.5)

    # HÀNG 2 - CỘT 2: Normalized ACF Khung 1
    ax_v_acf = fig.add_subplot(gs[1, 1])
    lag_ms_v = v_lags / fs * 1000
    ax_v_acf.plot(lag_ms_v, v_acfs, color="#2563eb", linewidth=1.4, label="Normalized ACF $R_{norm}(k)$")
    ax_v_acf.axhline(y=threshold, color="#15803d", linestyle=":", linewidth=1.3, label=f"Ngưỡng Voiced (T = {threshold:.4f})")
    
    v_is_voiced = (v_conf >= threshold)
    if v_is_voiced:
        ax_v_acf.axvline(x=v_t0_ms, color="#dc2626", linestyle="--", linewidth=1.5,
                         label=f"Đỉnh T0 = {v_t0_ms:.2f} ms\n=> F0 = {v_f0:.1f} Hz (R={v_conf:.3f} >= T)")
        ax_v_acf.plot([v_t0_ms], [v_conf], marker="o", markersize=7, color="#dc2626")
        title_v = f"Normalized ACF - Khung Hữu Thanh (Voiced)\nĐỉnh R_norm(T0) = {v_conf:.3f} >= T => HỮU THANH (F0 = {v_f0:.1f} Hz)"
        title_color_v = "#15803d"
    else:
        title_v = f"Normalized ACF - Khung Vô Thanh (Unvoiced)\nĐỉnh cực đại R_norm = {v_conf:.3f} < T => VÔ THANH (F0 = NaN)"
        title_color_v = "#b91c1c"

    ax_v_acf.set_title(title_v, fontsize=10.5, fontweight="bold", color=title_color_v)
    ax_v_acf.set_xlabel("Độ trễ lag (ms)", fontsize=10)
    ax_v_acf.set_ylabel("Độ tương quan $R_{norm}(k)$", fontsize=10)
    ax_v_acf.set_ylim([-0.3, 1.08])
    ax_v_acf.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax_v_acf.grid(True, linestyle="--", alpha=0.5)

    # 3. HÀNG 3 - CỘT 1: Waveform Khung 2
    ax_u_wave = fig.add_subplot(gs[2, 0])
    ax_u_wave.plot(u_time_ms, u_frame, color="#b91c1c", linewidth=1.0)
    wave_desc_u = "Dao động ngẫu nhiên dạng nhiễu, không tuần hoàn" if not u_is_voiced else f"Dao động tuần hoàn, T0 ≈ {u_t0_ms:.2f} ms"
    ax_u_wave.set_title(
        f"Khung {'Vô Thanh (Unvoiced)' if not u_is_voiced else 'Hữu Thanh (Voiced)'} [Index={idx_u}, t={u_t_center:.3f}s]\n"
        f"{wave_desc_u}",
        fontsize=10.5,
        fontweight="bold",
        color="#7f1d1d" if not u_is_voiced else "#1e3a8a",
    )
    ax_u_wave.set_xlabel("Thời gian trong khung (ms)", fontsize=10)
    ax_u_wave.set_ylabel("Biên độ", fontsize=10)
    ax_u_wave.grid(True, linestyle="--", alpha=0.5)

    # HÀNG 3 - CỘT 2: Normalized ACF Khung 2
    ax_u_acf = fig.add_subplot(gs[2, 1])
    lag_ms_u = u_lags / fs * 1000
    ax_u_acf.plot(lag_ms_u, u_acfs, color="#dc2626", linewidth=1.4, label="Normalized ACF $R_{norm}(k)$")
    ax_u_acf.axhline(y=threshold, color="#15803d", linestyle=":", linewidth=1.3, label=f"Ngưỡng Voiced (T = {threshold:.4f})")
    
    if u_is_voiced:
        ax_u_acf.axvline(x=u_t0_ms, color="#15803d", linestyle="--", linewidth=1.5,
                         label=f"Đỉnh T0 = {u_t0_ms:.2f} ms\n=> F0 = {u_f0:.1f} Hz (R={u_conf:.3f} >= T)")
        ax_u_acf.plot([u_t0_ms], [u_conf], marker="o", markersize=7, color="#15803d")
        title_u = f"Normalized ACF - Khung Hữu Thanh (Voiced)\nĐỉnh R_norm(T0) = {u_conf:.3f} >= T => HỮU THANH (F0 = {u_f0:.1f} Hz)"
        title_color_u = "#15803d"
    else:
        title_u = f"Normalized ACF - Khung Vô Thanh (Unvoiced)\nĐỉnh cực đại R_norm = {u_conf:.3f} < T => VÔ THANH (F0 = NaN)"
        title_color_u = "#b91c1c"

    ax_u_acf.set_title(title_u, fontsize=10.5, fontweight="bold", color=title_color_u)
    ax_u_acf.set_xlabel("Độ trễ lag (ms)", fontsize=10)
    ax_u_acf.set_ylabel("Độ tương quan $R_{norm}(k)$", fontsize=10)
    ax_u_acf.set_ylim([-0.3, 1.08])
    ax_u_acf.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax_u_acf.grid(True, linestyle="--", alpha=0.5)

    if output_path is None:
        output_path = Path(wav_path).parent.parent / "results" / f"two_frames_{wav_path.stem}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    info = {
        "wav_file": wav_path.name,
        "voiced": {
            "frame_idx": idx_v,
            "t_center_sec": v_t_center,
            "t0_ms": v_t0_ms,
            "f0_hz": v_f0,
            "peak_acf": v_conf,
            "is_voiced": True,
        },
        "unvoiced": {
            "frame_idx": idx_u,
            "t_center_sec": u_t_center,
            "peak_acf": u_conf,
            "is_voiced": u_is_voiced,
            "f0_hz": u_f0 if u_is_voiced else None,
        },
        "output_path": str(output_path),
    }
    return info