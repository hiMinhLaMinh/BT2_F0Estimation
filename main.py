"""
main.py - Điểm chạy duy nhất để đánh giá hiệu suất thuật toán trên tập Kiểm thử.

Quy trình:
1. Đọc ngưỡng tối ưu T và cấu hình từ results/optimal_threshold.txt (hoặc gán tay nếu file không có).
2. Duyệt qua 4 file .wav trong thư mục TinHieuKiemThu.
3. Ước lượng F0 (gọi hàm estimate_f0_track trong pitch_estimation.py).
4. Tính toán sai số định lượng (F0mean/F0std) qua evaluation.py.
5. Vẽ hình minh họa trực quan F0 Contour qua visualization.py.
"""

import sys
import numpy as np
from pathlib import Path

# Thêm đường dẫn để import các module tự viết
ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "TinHieuHuanLuyen"
RESULTS_DIR = ROOT / "results"

from io_utils import load_wav_normalized
from pitch_estimation import estimate_f0_track
from evaluation import evaluate_file, format_report
from visualization import plot_f0_contour

# --- 1. TẢI CẤU HÌNH TỪ QUÁ TRÌNH HUẤN LUYỆN ---
def load_optimal_config():
    """Đọc T, f0_min, f0_max, yin_threshold từ file optimal_threshold.txt"""
    config_file = RESULTS_DIR / "optimal_threshold.txt"
    
    # Cấu hình mặc định nếu không tìm thấy file
    config = {
        "f0_min": 70,
        "f0_max": 400,
        "yin_threshold": 0.1,
        "optimal_T": 0.522727 # Thay bằng giá trị bạn vừa tìm được
    }

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "F0 Range" in line:
                        parts = line.split(":")[-1].strip().split("-")
                        config["f0_min"] = int(parts[0].replace("Hz", "").strip())
                        config["f0_max"] = int(parts[1].replace("Hz", "").strip())
                    elif "YIN Threshold" in line:
                        config["yin_threshold"] = float(line.split(":")[-1].strip())
                    elif "NGƯỠNG TỐI ƯU (T)" in line:
                        config["optimal_T"] = float(line.split("=")[-1].strip())
            print(f"[INFO] Nạp cấu hình từ {config_file.name} thành công.")
        except Exception as e:
            print(f"[WARNING] Lỗi đọc {config_file.name}: {e}. Dùng cấu hình mặc định.")
    else:
        print("[WARNING] Không tìm thấy optimal_threshold.txt. Dùng cấu hình mặc định.")
        
    return config

# --- 2. CHƯƠNG TRÌNH CHÍNH ---
def main():
    if not TEST_DIR.exists():
        print(f"Lỗi: Không tìm thấy thư mục kiểm thử tại {TEST_DIR}")
        sys.exit(1)

    wav_files = sorted(TEST_DIR.glob("*.wav"))
    if not wav_files:
        print(f"Lỗi: Không tìm thấy file .wav nào trong {TEST_DIR}")
        sys.exit(1)

    config = load_optimal_config()
    print("-" * 50)
    print(f"CẤU HÌNH CHẠY KIỂM THỬ:")
    print(f"- Dải F0: {config['f0_min']} - {config['f0_max']} Hz")
    print(f"- Ngưỡng YIN: {config['yin_threshold']}")
    print(f"- Ngưỡng V/UV (T): {config['optimal_T']:.6f}")
    print("-" * 50)

    # Khởi tạo danh sách lưu kết quả evaluation
    all_results = []

    # Duyệt qua từng file kiểm thử (theo yêu cầu duyệt qua 4 file)
    for wav_path in wav_files:
        print(f"Đang xử lý file: {wav_path.name} ...")
        lab_path = wav_path.with_suffix(".lab")
        
        if not lab_path.exists():
            print(f"  -> [Bỏ qua] Không tìm thấy file {lab_path.name}")
            continue

        # 2.1. Đọc tín hiệu
        fs, signal = load_wav_normalized(wav_path)

        # 2.2. Ước lượng mảng F0 theo thời gian
        f0_track, centers_sec = estimate_f0_track(
            signal=signal,
            fs=fs,
            f0_min=config["f0_min"],
            f0_max=config["f0_max"],
            vuv_threshold=config["optimal_T"],
            yin_threshold=config["yin_threshold"]
        )

        # 2.3. Đánh giá sai số so với file .lab
        # Tham số min_num_f0, max_num_f0 có thể truyền tuỳ ý theo lời dặn của GV
        result = evaluate_file(f0_track, lab_path, min_num_f0=100, max_num_f0=1000)
        all_results.append(result)

        # 2.4. Vẽ đồ thị và lưu hình (Visualization)
        # Giả định visualization.py có hàm plot_f0_contour(wav_path, lab_path, f0_track, centers_sec, output_path)
        output_fig_path = RESULTS_DIR / f"{wav_path.stem}_contour.png"
        plot_f0_contour(wav_path, lab_path, f0_track, centers_sec, output_fig_path)

    # --- 3. IN BẢNG BÁO CÁO TỔNG KẾT ---
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP SAI SỐ TRÊN TẬP KIỂM THỬ")
    print("=" * 80)
    report_text = format_report(all_results)
    print(report_text)
    
    # Lưu bảng báo cáo ra file text để copy vào Word
    report_file = RESULTS_DIR / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[INFO] Đã lưu báo cáo sai số vào: {report_file}")
    print(f"[INFO] Đã lưu 4 hình ảnh F0 contour vào thư mục: {RESULTS_DIR}")

if __name__ == "__main__":
    main()