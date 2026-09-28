"""
Thu thập thống kê confidence (voiced vs unvoiced) trên tập huấn luyện.

Với mỗi khung: gán nhãn theo .lab (theo tâm khung), tính confidence bằng AMDF+YIN,
rồi tách thành nhóm voiced (nhãn 'v') và not-voiced (nhãn 'sil' hoặc 'uv').
Kết quả (meanV/stdV/meanU/stdU) in ra màn hình; mảng thô lưu vào results/*.npy
để đưa vào bước tìm ngưỡng (binary search / histogram từ BT1).

Lưu ý: 'sil' và 'uv' đang gộp chung. Muốn tách riêng, chia unvoiced_conf làm 2 list.

Chạy: python train.py
"""

from pathlib import Path

import numpy as np

from io_utils import label_at_time, load_wav_normalized, parse_lab_file
from pitch_estimation import f0_range_to_lag_range, frame_signal, short_time_amdf_yin

ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "TinHieuHuanLuyen"
RESULTS_DIR = ROOT / "results"


def collect_confidence_stats(train_dir, f0_min=70, f0_max=400, yin_threshold=0.1):
    train_dir = Path(train_dir)
    wav_files = sorted(train_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"Không tìm thấy file .wav trong {train_dir}")

    voiced_conf, unvoiced_conf = [], []
    n_skipped = 0

    for wav_path in wav_files:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists():
            print(f"[warn] thiếu .lab cho {wav_path.name}, bỏ qua file")
            continue

        fs, signal = load_wav_normalized(wav_path)
        segments, _, _ = parse_lab_file(lab_path)
        lag_min, lag_max = f0_range_to_lag_range(fs, f0_min, f0_max)
        frames, centers_sec = frame_signal(signal, fs)

        for frame, t_center in zip(frames, centers_sec):
            label = label_at_time(t_center, segments)
            if label is None:
                n_skipped += 1
                continue

            confidence = short_time_amdf_yin(frame, lag_min, lag_max, yin_threshold)[4]
            (voiced_conf if label == "v" else unvoiced_conf).append(confidence)

    voiced_conf = np.array(voiced_conf)
    unvoiced_conf = np.array(unvoiced_conf)

    def _stat(fn, arr):
        return fn(arr) if len(arr) else float("nan")

    stats = {
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
    stats, voiced_conf, unvoiced_conf = collect_confidence_stats(TRAIN_DIR)
    for key, val in stats.items():
        print(f"{key}: {val}")

    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / "voiced_confidence.npy", voiced_conf)
    np.save(RESULTS_DIR / "unvoiced_confidence.npy", unvoiced_conf)
    print(f"\nĐã lưu mảng confidence vào {RESULTS_DIR}")
