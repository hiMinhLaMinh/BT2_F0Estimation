"""Đọc file .wav và .lab (định dạng .lab: xem README.txt của giảng viên)."""

import numpy as np
from scipy.io import wavfile


def load_wav_normalized(wav_path):
    """Đọc .wav, trả về (fs, signal) với signal là float64 trong [-1, 1]."""
    fs, signal = wavfile.read(wav_path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # stereo -> mono

    if np.issubdtype(signal.dtype, np.integer):
        signal = signal.astype(np.float64) / np.iinfo(signal.dtype).max
    else:
        signal = signal.astype(np.float64)

    return fs, signal


def parse_lab_file(path):
    """
    Đọc file .lab.

    Returns:
        segments : list (start_sec, end_sec, label), label in {'sil','v','uv'}
        f0mean   : float hoặc None
        f0std    : float hoặc None
    """
    segments = []
    f0mean, f0std = None, None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) >= 3:
                try:
                    start, end = float(parts[0]), float(parts[1])
                    segments.append((start, end, parts[2]))
                    continue
                except ValueError:
                    pass

            if len(parts) == 2:
                key, val = parts
                if key.lower() == "f0mean":
                    f0mean = float(val)
                elif key.lower() == "f0std":
                    f0std = float(val)

    return segments, f0mean, f0std


def label_at_time(t_sec, segments):
    """Nhãn của đoạn chứa thời điểm t_sec, hoặc None nếu không đoạn nào chứa."""
    for start, end, label in segments:
        if start <= t_sec < end:
            return label
    return None
