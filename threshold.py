"""
Script tìm ngưỡng phân biệt Voiced / Unvoiced dựa trên thuật toán Binary Search.
Đọc các mảng confidence từ thư mục results/ và xuất kết quả ra màn hình & file text.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MAX_ITERATIONS = 100
THRESHOLD_TOLERANCE = 1e-6

TRAIN_F0_MIN = 70
TRAIN_F0_MAX = 400
METHOD = "acf"


def find_overlap_region(voiced_vals, unvoiced_vals):
    """Tìm vùng giao nhau (overlap) giữa phân bố Voiced và Unvoiced."""
    if len(voiced_vals) == 0 or len(unvoiced_vals) == 0:
        raise ValueError("Cần ít nhất 1 khung Voiced và 1 khung Unvoiced để tìm ngưỡng.")

    # Dùng phân vị 1% và 99% để loại bỏ khung ngoại lai
    min_v, max_v = float(np.percentile(voiced_vals, 1)), float(np.percentile(voiced_vals, 99))
    min_u, max_u = float(np.percentile(unvoiced_vals, 1)), float(np.percentile(unvoiced_vals, 99))

    ov_min = max(min_v, min_u)
    ov_max = min(max_v, max_u)
    has_overlap = ov_min < ov_max

    if not has_overlap:
        print(f"[Cảnh báo] Hai phân bố không giao nhau: "
              f"Voiced=[{min_v:.4f}, {max_v:.4f}], Unvoiced=[{min_u:.4f}, {max_u:.4f}]")
        ov_min = ov_max = (min(max_v, max_u) + max(min_v, min_u)) / 2.0

    v_overlap = voiced_vals[(voiced_vals >= ov_min) & (voiced_vals <= ov_max)]
    u_overlap = unvoiced_vals[(unvoiced_vals >= ov_min) & (unvoiced_vals <= ov_max)]

    return v_overlap, u_overlap, ov_min, ov_max, has_overlap


def compute_confusion(unvoiced_ov, voiced_ov, T):
    """
    Tính độ nhầm lẫn (confusion):
    - c_u: Giá trị unvoiced > T bị nhầm thành voiced.
    - c_v: Giá trị voiced < T bị nhầm thành unvoiced.
    """
    if len(unvoiced_ov) == 0 or len(voiced_ov) == 0:
        return 0.0, 0.0, 0.0
    c_u = float(np.mean(np.maximum(unvoiced_ov - T, 0.0)))
    c_v = float(np.mean(np.maximum(T - voiced_ov, 0.0)))
    return c_u, c_v, c_u - c_v


def binary_search_threshold(voiced_vals, unvoiced_vals):
    """Tìm kiếm nhị phân để chọn ngưỡng T tối ưu nhất."""
    v_ov, u_ov, tmin, tmax, has_overlap = find_overlap_region(voiced_vals, unvoiced_vals)

    if not has_overlap or len(v_ov) == 0 or len(u_ov) == 0:
        return tmin

    T = (tmin + tmax) / 2.0

    print(f"{'Iter':<5} | {'Tmin':<10} | {'Tmax':<10} | {'T (Ngưỡng)':<10} | {'Diff':<10}")
    print("-" * 55)

    for i in range(1, MAX_ITERATIONS + 1):
        c_u, c_v, diff = compute_confusion(u_ov, v_ov, T)

        print(f"{i:<5} | {tmin:<10.6f} | {tmax:<10.6f} | {T:<10.6f} | {diff:<10.6f}")

        if abs(diff) < THRESHOLD_TOLERANCE:
            print(f"-> Hội tụ sau {i} vòng lặp (Sai lệch diff ~ 0).")
            break
        if (tmax - tmin) < THRESHOLD_TOLERANCE:
            print(f"-> Dừng sau {i} vòng lặp do khoảng tìm kiếm [Tmin, Tmax] quá nhỏ.")
            break

        if diff > 0:
            tmin = T
        else:
            tmax = T

        T = (tmin + tmax) / 2.0

    return T


if __name__ == "__main__":
    v_path = RESULTS_DIR / "voiced_confidence.npy"
    u_path = RESULTS_DIR / "unvoiced_confidence.npy"

    if not v_path.exists() or not u_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {RESULTS_DIR}")
        print("Vui lòng chạy 'python train.py' trước để sinh dữ liệu.")
        sys.exit(1)

    print("Đang nạp dữ liệu confidence...")
    voiced_conf = np.load(v_path)
    unvoiced_conf = np.load(u_path)

    print(f"- Số khung Voiced: {len(voiced_conf)}")
    print(f"- Số khung Unvoiced: {len(unvoiced_conf)}")
    print(f"- Voiced range (1%-99%):   [{np.percentile(voiced_conf, 1):.4f}, {np.percentile(voiced_conf, 99):.4f}]")
    print(f"- Unvoiced range (1%-99%): [{np.percentile(unvoiced_conf, 1):.4f}, {np.percentile(unvoiced_conf, 99):.4f}]\n")

    print("Bắt đầu tìm kiếm ngưỡng phân biệt bằng Binary Search...")
    optimal_T = binary_search_threshold(voiced_conf, unvoiced_conf)

    print(f"\n[KẾT QUẢ] Ngưỡng (Threshold) tối ưu tìm được: {optimal_T:.6f}")

    false_voiced = int(np.sum(unvoiced_conf >= optimal_T))
    false_unvoiced = int(np.sum(voiced_conf < optimal_T))
    total_frames = len(voiced_conf) + len(unvoiced_conf)
    error_rate = (false_voiced + false_unvoiced) / total_frames * 100

    print(f"-> Tỷ lệ phân loại sai (Error Rate) trên tập huấn luyện: {error_rate:.2f}%")

    output_file = RESULTS_DIR / "optimal_threshold.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("KẾT QUẢ TÌM NGƯỠNG VOICED/UNVOICED (CONFIDENCE THRESHOLD)\n")
        f.write("=" * 60 + "\n")
        f.write("CẤU HÌNH THAM SỐ:\n")
        f.write(f"- Phương pháp (Method): {METHOD}\n")
        f.write(f"- Dải tần F0 (F0 Range): {TRAIN_F0_MIN} Hz - {TRAIN_F0_MAX} Hz\n")
        f.write("-" * 60 + "\n")
        f.write("SỐ LIỆU THỐNG KÊ:\n")
        f.write(f"- Số lượng khung Voiced:   {len(voiced_conf)}\n")
        f.write(f"- Số lượng khung Unvoiced: {len(unvoiced_conf)}\n")
        f.write(f"- Khung bị nhận diện nhầm: {false_voiced + false_unvoiced} frames\n")
        f.write(f"- Tỷ lệ phân loại sai:     {error_rate:.2f}%\n")
        f.write("-" * 60 + "\n")
        f.write(f"NGƯỠNG TỐI ƯU (T) = {optimal_T:.6f}\n")
        f.write("=" * 60 + "\n")
        f.write("Ghi chú: Giá trị confidence >= T sẽ được phân loại là Voiced, ngược lại là Unvoiced.\n")

    print(f"Đã lưu kết quả và cấu hình vào file: {output_file}")