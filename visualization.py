"""
visualization.py - Vẽ đồ thị Waveform và F0 Contour.
"""

import matplotlib.pyplot as plt
import numpy as np
from io_utils import load_wav_normalized, parse_lab_file

def plot_f0_contour(wav_path, lab_path, f0_track, centers_sec, output_path):
    """
    Vẽ 2 subplot:
    1. Waveform tín hiệu gốc.
    2. F0 Contour (đường F0 ước lượng + các mảng Voiced Ground Truth).
    """
    fs, signal = load_wav_normalized(wav_path)
    time_axis = np.arange(len(signal)) / fs

    segments, f0mean_gt, f0std_gt = parse_lab_file(lab_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Waveform
    ax1.plot(time_axis, signal, color='#2c3e50', linewidth=0.5)
    ax1.set_title(f"Waveform: {wav_path.name}")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: F0 Contour
    # Vẽ F0 ước lượng (bỏ qua các giá trị NaN)
    ax2.plot(centers_sec, f0_track, 'b.', markersize=4, label='F0 Ước lượng (AMDF+YIN)')
    
    # Vẽ nền làm nổi bật Ground Truth
    for start, end, label in segments:
        if label == 'v':
            ax2.axvspan(start, end, color='green', alpha=0.2, label='Ground Truth (Voiced)')
            # Để legend không lặp lại nhiều lần
            handles, labels = ax2.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys(), loc='upper right')

    ax2.set_title("Pitch Contour (F0)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim([50, 450]) # Dải F0
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()