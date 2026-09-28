"""
demo_two_frames.py - Minh họa trực quan hàm Normalized ACF cho 2 khung tín hiệu tự chọn theo yêu cầu mục 4 của đề bài.

Yêu cầu đề bài (Mục 4):
"Xuất hình vẽ kết quả hàm tự tương quan/AMDF của 2 khung tín hiệu (tự chọn để làm ví dụ minh họa)
và nêu rõ vì sao hàm này giúp tìm F0:
● 1 khung chứa tiếng nói tuần hoàn (hữu thanh) F0 = ? (Hz)
● 1 khung chứa tiếng nói không tuần hoàn (vô thanh) F0 không xác định."

Cách sử dụng:
1. Chạy mặc định (chọn 2 khung tiêu biểu của studio_M1.wav):
   python demo_two_frames.py

2. Tự chọn thời điểm (giây) cho khung hữu thanh (--t_v) và vô thanh (--t_u):
   python demo_two_frames.py --file data/TinHieuHuanLuyen/studio_M1.wav --t_v 1.05 --t_u 1.62

3. Tự chọn theo chỉ số khung (frame index):
   python demo_two_frames.py --file data/TinHieuHuanLuyen/phone_F1.wav --frame_v 80 --frame_u 140

4. Chỉ định file ảnh xuất ra:
   python demo_two_frames.py --output results/my_two_frames.png
"""

import argparse
from pathlib import Path
from visualization import plot_two_frames_acf

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def get_optimal_threshold():
    cfg_file = RESULTS_DIR / "optimal_threshold.txt"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "NGƯỠNG TỐI ƯU (T)" in line:
                        return float(line.split("=")[-1].strip())
        except Exception:
            pass
    return 0.591812


def main():
    parser = argparse.ArgumentParser(
        description="Minh họa trực quan Normalized ACF cho 2 khung tự chọn (Hữu thanh vs Vô thanh)."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=str(ROOT / "data" / "TinHieuHuanLuyen" / "studio_M1.wav"),
        help="Đường dẫn đến file .wav cần khảo sát (mặc định: studio_M1.wav)",
    )
    parser.add_argument(
        "--t_v",
        type=float,
        default=1.05,
        help="Thời điểm (giây) của khung hữu thanh - Voiced (mặc định: 1.05s)",
    )
    parser.add_argument(
        "--t_u",
        type=float,
        default=1.62,
        help="Thời điểm (giây) của khung vô thanh - Unvoiced (mặc định: 1.62s)",
    )
    parser.add_argument(
        "--frame_v",
        type=int,
        default=None,
        help="Chỉ số khung (index) của khung hữu thanh (ưu tiên hơn --t_v)",
    )
    parser.add_argument(
        "--frame_u",
        type=int,
        default=None,
        help="Chỉ số khung (index) của khung vô thanh (ưu tiên hơn --t_u)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(RESULTS_DIR / "two_frames_illustration.png"),
        help="Đường dẫn file ảnh xuất ra (mặc định: results/two_frames_illustration.png)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Ngưỡng phân biệt Voiced/Unvoiced (mặc định: đọc từ results/optimal_threshold.txt)",
    )

    args = parser.parse_args()

    threshold = args.threshold if args.threshold is not None else get_optimal_threshold()
    wav_path = Path(args.file)

    if not wav_path.exists():
        print(f"Lỗi: Không tìm thấy file âm thanh tại {wav_path}")
        return

    print("=" * 70)
    print("MINH HỌA TRỰC QUAN 2 KHUNG TÍN HIỆU TỰ CHỌN (NORMALIZED ACF)")
    print("=" * 70)
    print(f"- File âm thanh: {wav_path.name}")
    print(f"- Ngưỡng phân loại V/UV (T): {threshold:.6f}")
    if args.frame_v is not None:
        print(f"- Khung Hữu thanh (Voiced): Frame Index = {args.frame_v}")
    else:
        print(f"- Khung Hữu thanh (Voiced): Thời điểm target = {args.t_v}s")

    if args.frame_u is not None:
        print(f"- Khung Vô thanh (Unvoiced): Frame Index = {args.frame_u}")
    else:
        print(f"- Khung Vô thanh (Unvoiced): Thời điểm target = {args.t_u}s")
    print("-" * 70)

    # Gọi hàm vẽ đồ thị
    info = plot_two_frames_acf(
        wav_path=wav_path,
        t_voiced=args.t_v,
        t_unvoiced=args.t_u,
        frame_idx_v=args.frame_v,
        frame_idx_u=args.frame_u,
        output_path=args.output,
        threshold=threshold,
    )

    v = info["voiced"]
    u = info["unvoiced"]

    # In kết quả chi tiết
    print("\n[KẾT QUẢ PHÂN TÍCH 2 KHUNG]:")
    print(f"1. Khung 1 (Target Voiced):")
    print(f"   - Tâm khung: t = {v['t_center_sec']:.3f} s (Frame Index: {v['frame_idx']})")
    if v['is_voiced']:
        print(f"   - Biên độ đỉnh tương quan cực đại: R_norm = {v['peak_acf']:.4f} (>= {threshold:.4f} => HỮU THANH)")
        print(f"   - Chu kỳ cơ bản T0 tìm được:       T0 = {v['t0_ms']:.2f} ms")
        print(f"   - Tần số cơ bản F0 = 1 / T0:       F0 = {v['f0_hz']:.1f} Hz")
    else:
        print(f"   - Biên độ đỉnh tương quan cực đại: R_norm = {v['peak_acf']:.4f} (< {threshold:.4f} => VÔ THANH)")
        print(f"   - Tần số cơ bản F0:                KHÔNG XÁC ĐỊNH (NaN)")

    print()
    print(f"2. Khung 2 (Target Unvoiced):")
    print(f"   - Tâm khung: t = {u['t_center_sec']:.3f} s (Frame Index: {u['frame_idx']})")
    if not u['is_voiced']:
        print(f"   - Biên độ đỉnh tương quan cực đại: R_norm = {u['peak_acf']:.4f} (< {threshold:.4f} => VÔ THANH)")
        print(f"   - Tần số cơ bản F0:                KHÔNG XÁC ĐỊNH (NaN)")
    else:
        print(f"   - Biên độ đỉnh tương quan cực đại: R_norm = {u['peak_acf']:.4f} (>= {threshold:.4f} => HỮU THANH)")
        print(f"   - Chu kỳ cơ bản T0 tìm được:       T0 = {u['t0_ms']:.2f} ms")
        print(f"   - Tần số cơ bản F0 = 1 / T0:       F0 = {u['f0_hz']:.1f} Hz")
    print("=" * 70)

    # Lưu nội dung giải thích lý thuyết ra file text để SV copy vào báo cáo Word/Slide
    explanation_file = RESULTS_DIR / "two_frames_explanation.txt"
    with open(explanation_file, "w", encoding="utf-8") as f:
        f.write("GIẢI THÍCH NGUYÊN LÝ HÀM TỰ TƯƠNG QUAN (ACF) TÌM F0 (MỤC 4 ĐỀ BÀI)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"1. Khung 1 (Hữu Thanh - Voiced) - File: {wav_path.name}, t = {v['t_center_sec']:.3f}s:\n")
        if v['is_voiced']:
            f.write(f"   - Dạng sóng: Tín hiệu thể hiện các dao động tuần hoàn điều hòa của dây thanh âm lặp lại liên tục.\n")
            f.write(f"   - Hàm tự tương quan ACF: Khi dịch chuyển một độ trễ đúng bằng chu kỳ dao động cơ bản k = T0,\n")
            f.write(f"     dạng sóng tự trùng khớp pha với chính nó x(n) ≈ x(n + T0).\n")
            f.write(f"     Do đó, tích phân vô hướng chuẩn hóa đạt giá trị cực đại nổi bật: R_norm(T0) = {v['peak_acf']:.4f} >= {threshold:.4f}.\n")
            f.write(f"   - Xác định F0: Chu kỳ T0 = {v['t0_ms']:.2f} ms => Tần số cơ bản F0 = 1 / T0 = {v['f0_hz']:.1f} Hz.\n\n")
        else:
            f.write(f"   - Dạng sóng: Tín hiệu vô thanh / khoảng lặng, không có tính tuần hoàn.\n")
            f.write(f"   - Hàm tự tương quan ACF: R_norm = {v['peak_acf']:.4f} < {threshold:.4f} (dưới ngưỡng T).\n")
            f.write(f"   - Xác định F0: KHÔNG XÁC ĐỊNH (NaN).\n\n")

        f.write(f"2. Khung 2 (Vô Thanh - Unvoiced) - File: {wav_path.name}, t = {u['t_center_sec']:.3f}s:\n")
        if not u['is_voiced']:
            f.write(f"   - Dạng sóng: Âm vô thanh phát sinh do luồng khí hỗn loạn qua khe hẹp thanh quản, dạng sóng ngẫu nhiên,\n")
            f.write(f"     biến thiên hỗn loạn như nhiễu trắng, không có tính chu kỳ tuần hoàn.\n")
            f.write(f"   - Hàm tự tương quan ACF: Với tín hiệu ngẫu nhiên, các mẫu tín hiệu tại các khoảng dịch k khác nhau không có\n")
            f.write(f"     mối tương quan đồng pha, tích phân nhân chập triệt tiêu lẫn nhau. Giá trị R_norm(k) dao động nhỏ quanh 0.\n")
            f.write(f"     Biên độ lớn nhất chỉ đạt {u['peak_acf']:.4f} < {threshold:.4f} (dưới ngưỡng Voiced T).\n")
            f.write(f"   - Kết luận: Không tồn tại chu kỳ lặp lại => Khung là Vô thanh, F0 KHÔNG XÁC ĐỊNH (NaN).\n\n")
        else:
            f.write(f"   - Dạng sóng: Tín hiệu có tính tuần hoàn lặp lại (chu kỳ T0 ≈ {u['t0_ms']:.2f} ms).\n")
            f.write(f"   - Hàm tự tương quan ACF: R_norm = {u['peak_acf']:.4f} >= {threshold:.4f} (trên ngưỡng T).\n")
            f.write(f"   - Xác định F0: F0 = {u['f0_hz']:.1f} Hz.\n\n")

        f.write("3. Kết luận chung về vai trò của hàm ACF:\n")
        f.write("   Hàm tự tương quan chuẩn hóa (Normalized ACF) giải quyết được cả 2 nhiệm vụ cốt lõi:\n")
        f.write("   - Phân biệt Voiced / Unvoiced: Dựa vào độ lớn của đỉnh cực đại so với ngưỡng tối ưu T.\n")
        f.write("   - Ước lượng F0: Khi là Voiced, vị trí đỉnh cực đại nổi bật đầu tiên chính là chu kỳ T0 => F0 = fs / lag = 1 / T0.\n")

    print(f"[INFO] Đã lưu hình minh họa trực quan tại: {info['output_path']}")
    print(f"[INFO] Đã lưu bản giải thích lý thuyết chi tiết tại: {explanation_file}")


if __name__ == "__main__":
    main()
