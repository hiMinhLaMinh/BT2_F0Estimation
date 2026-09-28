"""
Thu thập thống kê confidence (Voiced vs Unvoiced) trên tập huấn luyện (TinHieuHuanLuyen).

Hỗ trợ 2 phương pháp:
- 'acf' (Khuyến nghị): Normalized Autocorrelation Function (ACF)
- 'amdf_yin': AMDF kết hợp YIN

Với mỗi khung: gán nhãn theo file .lab (tại tâm khung), tính confidence:
- Với ACF: confidence là biên độ đỉnh cực đại địa phương của normalized ACF (nằm trong [0, 1]).
- Tách thành nhóm Voiced (nhãn 'v') và Unvoiced (nhãn 'uv' hoặc 'sil').

In ra màn hình meanV, stdV, meanU, stdU và lưu mảng thô vào results/*.npy
để đưa vào bước tìm ngưỡng tối ưu (threshold.py bằng Binary Search).

Chạy: python train.py
"""

from pathlib import Path
import numpy as np

from io_utils import label_at_time, load_wav_normalized, parse_lab_file
from pitch_estimation import (
    f0_range_to_lag_range,
    frame_signal,
    short_time_acf_normalized,
    short_time_amdf_yin,
)

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "TinHieuHuanLuyen"
RESULTS_DIR = ROOT / "results"


def collect_confidence_stats(train_dir, f0_min=70, f0_max=400, method="acf", yin_threshold=0.1):
    train_dir = Path(train_dir)
    wav_files = sorted(train_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Không tìm thấy file .wav trong {train_dir}")

    voiced_conf, unvoiced_conf = [], []
    n_skipped = 0

    print(f"Bắt đầu thu thập thống kê trên tập huấn luyện ({train_dir.name}) bằng phương pháp: {method.upper()}...")

    for wav_path in wav_files:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            print(f"[warn] thiếu .lab cho {wav_path.name}, bỏ qua file")
            continue

        fs, signal = load_wav_normalized(wav_path)
        segments, _, _ = parse_lab_file(lab_path)
        lag_min, lag_max = f0_range_to_lag_range(fs, f0_min, f0_max)
        frames, centers_sec = frame_signal(signal, fs)

        # Áp dụng bộ lọc năng lượng ngắn hạn (Short-Time Energy) đồng bộ với pitch_estimation.py
        energies = np.array([float(np.mean((fr - np.mean(fr)) ** 2)) for fr in frames])
        sil_thresh = max(1e-5, float(np.max(energies)) * 0.005)

        for frame, t_center, e in zip(frames, centers_sec, energies):
            label = label_at_time(t_center, segments)
            if label is None:
                n_skipped += 1
                continue

            # Loại bỏ các khung khoảng lặng (silence) để phân bố Unvoiced phản ánh chính xác âm vô thanh thực tế
            if e < sil_thresh:
                n_skipped += 1
                continue

            if method == "acf":
                # Normalized ACF: confidence là biên độ đỉnh tương quan cực đại
                _, _, _, confidence, _ = short_time_acf_normalized(frame, lag_min, lag_max, threshold=0.0)
            elif method == "amdf_yin":
                confidence = short_time_amdf_yin(frame, lag_min, lag_max, yin_threshold)[4]
            else:
                raise ValueError(f"Phương pháp không hợp lệ: {method}")

            if label == "v":
                voiced_conf.append(confidence)
            else:
                unvoiced_conf.append(confidence)

    voiced_conf = np.array(voiced_conf, dtype=np.float64)
    unvoiced_conf = np.array(unvoiced_conf, dtype=np.float64)

    def _stat(fn, arr):
        return float(fn(arr)) if len(arr) else float("nan")

    stats = {
        "method": method,
        "meanV": _stat(np.mean, voiced_conf),
        "stdV": _stat(np.std, voiced_conf),
        "meanU": _stat(np.mean, unvoiced_conf),
        "stdU": _stat(np.std, unvoiced_conf),
        "n_voiced_frames": len(voiced_conf),
        "n_unvoiced_frames": len(unvoiced_conf),
        "n_skipped_frames": n_skipped,
    }
    return stats, voiced_conf, unvoiced_conf


if __name__ == "__main__":
    METHOD = "acf"  # Chuyển sang Normalized ACF
    stats, voiced_conf, unvoiced_conf = collect_confidence_stats(TRAIN_DIR, method=METHOD)

    print("=" * 60)
    print(f"KẾT QUẢ THỐNG KÊ PHÂN BỐ ({METHOD.upper()}):")
    print(f"- Voiced frames:   {stats['n_voiced_frames']} khung")
    print(f"  meanV = {stats['meanV']:.4f}, stdV = {stats['stdV']:.4f}")
    print(f"- Unvoiced frames: {stats['n_unvoiced_frames']} khung")
    print(f"  meanU = {stats['meanU']:.4f}, stdU = {stats['stdU']:.4f}")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / "voiced_confidence.npy", voiced_conf)
    np.save(RESULTS_DIR / "unvoiced_confidence.npy", unvoiced_conf)
    print(f"Đã lưu mảng confidence vào {RESULTS_DIR}")
