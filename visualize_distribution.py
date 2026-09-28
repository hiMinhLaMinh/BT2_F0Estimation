"""
Script trực quan hóa phân bố Confidence của Voiced và Unvoiced.
Giúp quan sát rõ vùng chồng lấp (overlap) giữa hai lớp dữ liệu.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Cấu hình đường dẫn
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

def plot_confidence_distribution(optimal_threshold=None):
    v_path = RESULTS_DIR / "voiced_confidence.npy"
    u_path = RESULTS_DIR / "unvoiced_confidence.npy"

    if not v_path.exists() or not u_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {RESULTS_DIR}")
        print("Vui lòng chạy 'python train.py' trước để sinh dữ liệu.")
        return

    # 1. Nạp dữ liệu
    print("Đang nạp dữ liệu confidence...")
    voiced_conf = np.load(v_path)
    unvoiced_conf = np.load(u_path)

    # 2. Khởi tạo biểu đồ
    plt.figure(figsize=(10, 6))

    # Vẽ histogram cho Unvoiced (Màu đỏ, trong suốt 60%)
    plt.hist(unvoiced_conf, bins=50, alpha=0.6, color='red', 
             label=f'Unvoiced (n={len(unvoiced_conf)})', edgecolor='black', linewidth=0.5)

    # Vẽ histogram cho Voiced (Màu xanh, trong suốt 60%)
    plt.hist(voiced_conf, bins=50, alpha=0.6, color='blue', 
             label=f'Voiced (n={len(voiced_conf)})', edgecolor='black', linewidth=0.5)

    # 3. Vẽ đường ranh giới Ngưỡng T (Nếu có cung cấp)
    if optimal_threshold is not None:
        plt.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2,
                    label=f'Optimal T = {optimal_threshold:.4f}')

    # 4. Trang trí biểu đồ
    plt.title('Phân Bố Confidence: Voiced vs Unvoiced', fontsize=14, fontweight='bold')
    plt.xlabel('Confidence Value', fontsize=12)
    plt.ylabel('Số lượng Khung (Frames)', fontsize=12)
    
    # Thiết lập giới hạn trục X từ min đến max của toàn bộ dữ liệu
    all_data = np.concatenate([voiced_conf, unvoiced_conf])
    plt.xlim(max(0.0, np.min(all_data) - 0.05), min(1.0, np.max(all_data) + 0.05))
    
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # 5. Lưu và hiển thị
    out_file = RESULTS_DIR / "confidence_histogram.png"
    plt.savefig(out_file, dpi=300)
    print(f"Đã lưu biểu đồ tại: {out_file}")
    plt.show()

if __name__ == "__main__":
    # Bạn có thể điền giá trị T tối ưu tìm được từ threshold.py vào đây
    # Ví dụ: T = 0.522727
    T_OPTIMAL = 0.522727 
    
    plot_confidence_distribution(optimal_threshold=T_OPTIMAL)