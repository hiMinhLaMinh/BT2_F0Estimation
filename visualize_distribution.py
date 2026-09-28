"""
Script trực quan hóa phân bố Confidence của Voiced và Unvoiced.
Giúp quan sát rõ vùng chồng lấp (overlap) giữa hai lớp dữ liệu và ngưỡng tối ưu T.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def get_optimal_threshold_from_file():
    cfg_file = RESULTS_DIR / "optimal_threshold.txt"
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            for line in f:
                if "NGƯỠNG TỐI ƯU (T)" in line:
                    return float(line.split("=")[-1].strip())
    return None


def plot_confidence_distribution(optimal_threshold=None):
    v_path = RESULTS_DIR / "voiced_confidence.npy"
    u_path = RESULTS_DIR / "unvoiced_confidence.npy"

    if not v_path.exists() or not u_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {RESULTS_DIR}")
        print("Vui lòng chạy 'python train.py' trước để sinh dữ liệu.")
        return

    print("Đang nạp dữ liệu confidence...")
    voiced_conf = np.load(v_path)
    unvoiced_conf = np.load(u_path)

    if optimal_threshold is None:
        optimal_threshold = get_optimal_threshold_from_file()

    plt.figure(figsize=(11, 6))

    # Vẽ histogram cho Unvoiced (Đỏ) và Voiced (Xanh dương)
    plt.hist(
        unvoiced_conf,
        bins=50,
        alpha=0.55,
        color="#ef4444",
        label=f"Unvoiced (n={len(unvoiced_conf)}, μ={unvoiced_conf.mean():.3f})",
        edgecolor="black",
        linewidth=0.5,
    )

    plt.hist(
        voiced_conf,
        bins=50,
        alpha=0.55,
        color="#3b82f6",
        label=f"Voiced (n={len(voiced_conf)}, μ={voiced_conf.mean():.3f})",
        edgecolor="black",
        linewidth=0.5,
    )

    if optimal_threshold is not None:
        plt.axvline(
            x=optimal_threshold,
            color="#15803d",
            linestyle="--",
            linewidth=2.2,
            label=f"Optimal T = {optimal_threshold:.4f}",
        )

    plt.title("Phân Bố Độ Tương Quan (Normalized ACF Peak Confidence): Voiced vs Unvoiced", fontsize=13, fontweight="bold")
    plt.xlabel("Biên độ cực đại Normalized ACF", fontsize=11)
    plt.ylabel("Số lượng khung (Frames)", fontsize=11)

    all_data = np.concatenate([voiced_conf, unvoiced_conf])
    plt.xlim(max(0.0, float(np.min(all_data)) - 0.05), min(1.0, float(np.max(all_data)) + 0.05))

    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_file = RESULTS_DIR / "confidence_histogram.png"
    plt.savefig(out_file, dpi=300)
    print(f"Đã lưu biểu đồ phân bố tại: {out_file}")
    plt.close()


if __name__ == "__main__":
    plot_confidence_distribution()