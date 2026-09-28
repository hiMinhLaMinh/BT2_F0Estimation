"""
main.py - Điểm chạy duy nhất để đánh giá hiệu suất thuật toán trên tập Kiểm thử (TinHieuKiemThu).

Quy trình:
1. Đọc ngưỡng tối ưu T và cấu hình từ results/optimal_threshold.txt.
2. Duyệt qua 4 file .wav trong thư mục TinHieuKiemThu (phone_F2, phone_M2, studio_F2, studio_M2).
3. Ước lượng F0 (gọi hàm estimate_f0_track bằng Normalized ACF).
4. Tính toán sai số định lượng (F0mean/F0std) so với file .lab qua evaluation.py.
5. Xuất 4 figure thể hiện input waveform & output F0 contour qua visualization.py.
6. In bảng tổng hợp sai số và lưu kết quả báo cáo.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "TinHieuKiemThu"
RESULTS_DIR = ROOT / "results"

from io_utils import load_wav_normalized
from pitch_estimation import estimate_f0_track
from evaluation import evaluate_file, format_report
from visualization import plot_f0_contour


def load_optimal_config():
    """Đọc T, dải F0, phương pháp từ file optimal_threshold.txt"""
    config_file = RESULTS_DIR / "optimal_threshold.txt"

    config = {
        "method": "acf",
        "f0_min": 70,
        "f0_max": 400,
        "optimal_T": 0.591812,
        "yin_threshold": 0.1,
    }

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "Method" in line or "Phương pháp" in line:
                        config["method"] = line.split(":")[-1].strip().lower()
                    elif "F0 Range" in line or "Dải tần F0" in line:
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


def main():
    if not TEST_DIR.exists():
        print(f"Lỗi: Không tìm thấy thư mục kiểm thử tại {TEST_DIR}")
        sys.exit(1)

    wav_files = sorted(TEST_DIR.glob("*.wav"))
    if not wav_files:
        print(f"Lỗi: Không tìm thấy file .wav nào trong {TEST_DIR}")
        sys.exit(1)

    config = load_optimal_config()
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 65)
    print("CHẠY KIỂM THỬ TRÊN TẬP TÍN HIỆU KIỂM THỬ (TinHieuKiemThu):")
    print(f"- Phương pháp: {config['method'].upper()} (Normalized Autocorrelation)")
    print(f"- Dải F0 khảo sát: {config['f0_min']} Hz - {config['f0_max']} Hz")
    print(f"- Ngưỡng tối ưu V/UV (T): {config['optimal_T']:.6f}")
    print("=" * 65)

    all_results = []

    for wav_path in wav_files:
        print(f"-> Đang xử lý file: {wav_path.name} ...")
        lab_path = wav_path.with_suffix(".lab")

        if not lab_path.exists():
            print(f"  [Bỏ qua] Không tìm thấy file nhãn chuẩn {lab_path.name}")
            continue

        # 1. Đọc tín hiệu
        fs, signal = load_wav_normalized(wav_path)

        # 2. Ước lượng F0 theo thời gian
        f0_track, centers_sec = estimate_f0_track(
            signal=signal,
            fs=fs,
            f0_min=config["f0_min"],
            f0_max=config["f0_max"],
            vuv_threshold=config["optimal_T"],
            yin_threshold=config["yin_threshold"],
            method=config["method"],
            apply_median_filter=True,
        )

        # 3. Đánh giá sai số so với file .lab
        result = evaluate_file(f0_track, lab_path, min_num_f0=50, max_num_f0=500)
        all_results.append(result)

        # 4. Xuất figure minh họa input waveform & output F0 contour
        output_fig_path = RESULTS_DIR / f"{wav_path.stem}_contour.png"
        method_label = "Normalized ACF" if config["method"] == "acf" else "AMDF+YIN"
        plot_f0_contour(wav_path, lab_path, f0_track, centers_sec, output_fig_path, method_name=method_label)
        print(f"   Đã lưu figure: {output_fig_path.name}")

    # 5. In và lưu bảng báo cáo
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP SAI SỐ TRÊN TẬP KIỂM THỬ (TinHieuKiemThu)")
    print("=" * 80)
    report_text = format_report(all_results)
    print(report_text)

    report_file = RESULTS_DIR / "test_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("BẢNG TỔNG HỢP SAI SỐ TRÊN TẬP KIỂM THỬ (TinHieuKiemThu)\n")
        f.write(f"Phương pháp: {config['method'].upper()}\n")
        f.write(f"Ngưỡng T: {config['optimal_T']:.6f}\n")
        f.write("-" * 80 + "\n")
        f.write(report_text + "\n")

    print(f"\n[INFO] Đã lưu báo cáo sai số vào: {report_file}")
    print(f"[INFO] Đã hoàn thành xuất các figure vào: {RESULTS_DIR}")


if __name__ == "__main__":
    main()