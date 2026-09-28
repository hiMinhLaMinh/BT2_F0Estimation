# BT2 – Tìm tần số cơ bản (F0) của tín hiệu tiếng nói

Thuật toán chính: **Normalized Autocorrelation Function (ACF)** kết hợp:
- Chuẩn hóa Pearson không thiên lệch theo độ trễ lag $k$.
- Thuật toán chọn đỉnh cực đại địa phương nổi bật đầu tiên (**first prominent peak**) để chống lỗi chia đôi tần số (**pitch halving / bắt nhầm $2T_0$**).
- Nội suy parabol quanh đỉnh cực đại để đạt độ chính xác dưới mẫu (**sub-sample accuracy**).
- Bộ lọc năng lượng (**Short-Time Energy Gating**) loại bỏ triệt để điểm ảo trong khoảng lặng (**silence**).
- Bộ lọc trung vị 5 điểm (**5-point Median Filter**) làm mượt đường pitch contour tự nhiên, triệt tiêu lỗi đột biến 1-2 khung.

> **Ghi chú:** Kết quả và mã nguồn cũ của phương pháp AMDF-YIN vẫn được lưu trữ nguyên vẹn trong thư mục `results_amdf_yin/` và trong `pitch_estimation.py`.

---

## Ràng buộc môn học
- **Thư viện xử lý tín hiệu:** Chỉ dùng `numpy` và `scipy.io.wavfile` (`matplotlib` chỉ dùng để xuất đồ thị).
- **Tham số chuẩn hóa để xếp hạng:** 
  - Độ dài khung (**Frame length**): `25 ms`
  - Độ dịch khung (**Frame shift**): `10 ms`
  - Dải tần số $F_0$ khảo sát: `70 – 400 Hz`

---

## Cấu trúc thư mục

```
BT2_F0Estimation/
├── main.py                     # Điểm chạy duy nhất: duyệt 4 file test, xuất 4 figure & báo cáo
├── train.py                    # Thu thập thống kê phân bố confidence Voiced/Unvoiced trên tập huấn luyện
├── threshold.py                # Tìm ngưỡng tối ưu T bằng Binary Search
├── demo_two_frames.py          # Xuất hình minh họa 2 khung tín hiệu (Voiced vs Unvoiced) theo mục 4 đề bài
├── survey_frame_length.py      # Khảo sát ảnh hưởng độ dài khung (20ms, 25ms, 30ms) theo yêu cầu đề bài
├── pitch_estimation.py         # Lõi thuật toán: framing, Normalized ACF, AMDF, AMDF+YIN
├── evaluation.py               # Đánh giá sai số F0mean, F0std, độ lệch % và số lượng F0
├── visualization.py            # Vẽ đồ thị Waveform và F0 Contour kèm Ground Truth
├── io_utils.py                 # Đọc và chuẩn hóa file .wav, phân tích cú pháp file .lab
├── instruction/                # Đề bài PDF của GV và tệp quan trọng
├── data/
│   ├── TinHieuHuanLuyen/       # 4 file huấn luyện: phone_F1, phone_M1, studio_F1, studio_M1 (.wav + .lab)
│   └── TinHieuKiemThu/         # 4 file kiểm thử: phone_F2, phone_M2, studio_F2, studio_M2 (.wav + .lab)
├── results/                    # Kết quả chạy thuật toán Normalized ACF:
│   ├── optimal_threshold.txt   # Cấu hình tham số và ngưỡng T tối ưu (T ≈ 0.5918)
│   ├── test_evaluation_report.txt # Bảng báo cáo sai số trên tập kiểm thử
│   ├── frame_length_survey.txt # Bảng khảo sát ảnh hưởng độ dài khung (20ms vs 25ms vs 30ms)
│   ├── confidence_histogram.png# Biểu đồ phân bố confidence và ngưỡng T
│   ├── two_frames_illustration.png # Hình minh họa 2 khung tín hiệu Voiced & Unvoiced
│   └── *_contour.png           # 4 hình vẽ pitch contour cho 4 file kiểm thử
├── results_amdf_yin/           # Lưu trữ toàn bộ kết quả của bản chạy AMDF-YIN trước đó
└── requirements.txt
```

---

## Kết quả đánh giá trên tập kiểm thử (`TinHieuKiemThu`)

```
================================================================================
BẢNG TỔNG HỢP SAI SỐ TRÊN TẬP KIỂM THỬ (TinHieuKiemThu)
================================================================================
File             F0mean  (chuẩn)    lệch   lệch%    F0std  (chuẩn)    lệch   lệch%   #F0
----------------------------------------------------------------------------------------
phone_F2          148.0    145.0    +3.0    +2.1     35.0     33.7    +1.3    +3.8   201
phone_M2          130.4    129.0    +1.4    +1.1     17.2     18.6    -1.4    -7.5   123
studio_F2         200.1    200.0    +0.1    +0.1     44.8     46.1    -1.3    -2.8   131
studio_M2         155.1    155.0    +0.1    +0.0     30.9     30.8    +0.1    +0.3   117
----------------------------------------------------------------------------------------
Trung bình |lệch| trên 4/4 file: F0mean 1.16 Hz, F0std 1.01 Hz
```

---

## Hướng dẫn chạy chương trình

Chạy tuần tự các bước sau từ thư mục `BT2_F0Estimation/`:

1. **Bước 1: Huấn luyện và thu thập phân bố Voiced / Unvoiced**
   ```bash
   python train.py
   ```
2. **Bước 2: Tìm ngưỡng phân biệt Voiced / Unvoiced bằng Binary Search**
   ```bash
   python threshold.py
   ```
3. **Bước 3: Chạy kiểm thử, xuất 4 figure và in bảng tổng hợp sai số**
   ```bash
   python main.py
   ```
4. **Bước 4: Sinh các đồ thị phục vụ báo cáo / slide thuyết trình**
   ```bash
   python visualize_distribution.py   # Biểu đồ phân bố histogram
   python demo_two_frames.py          # Minh họa 2 khung tín hiệu Voiced vs Unvoiced
   python survey_frame_length.py      # Khảo sát độ dài khung 20ms, 25ms, 30ms
   ```
