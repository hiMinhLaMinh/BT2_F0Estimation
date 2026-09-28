# BT2 – Tìm tần số cơ bản (F0) của tín hiệu tiếng nói

Cài đặt AMDF (baseline + cải tiến YIN) và ACF baseline; thu thập thống kê để tìm ngưỡng voiced/unvoiced.

**Ràng buộc:** chỉ dùng `numpy` và `scipy.io.wavfile` cho xử lý tín hiệu (`matplotlib` chỉ để vẽ hình).
**Cố định để xếp hạng:** frame length 25 ms, frame shift 10 ms, dải F0 tìm kiếm 70–400 Hz.

## Cấu trúc thư mục

```
BT2_F0Estimation/
├── main.py               # (chưa viết) điểm chạy duy nhất, xuất figure cho 4 file test
├── train.py              # thống kê confidence voiced/unvoiced trên tập huấn luyện
├── io_utils.py           # đọc .wav và .lab
├── pitch_estimation.py   # lõi thuật toán: framing, ACF, AMDF, AMDF+YIN
├── evaluation.py         # (chưa viết) sai số F0mean/F0std, số lượng F0 so với .lab
├── visualization.py      # (chưa viết) vẽ hình theo yêu cầu đề bài
├── data/
│   ├── TinHieuHuanLuyen/ # file huấn luyện (.wav + .lab)
│   └── TinHieuKiemThu/   # file kiểm thử (.wav + .lab)
├── results/              # (sinh ra) voiced_confidence.npy, unvoiced_confidence.npy
└── requirements.txt
```

## Mô tả các file

**`io_utils.py`**: `load_wav_normalized` (đọc wav về float trong [-1, 1]), `parse_lab_file` (đọc segments và F0mean/F0std), `label_at_time` (nhãn của một thời điểm).

**`pitch_estimation.py`**: hàm tính trên một khung, không đọc file.

| Hàm | Mục đích |
|---|---|
| `frame_signal` | Chia khung 25/10 ms, trả về khung và thời điểm tâm khung |
| `f0_range_to_lag_range` | Đổi dải F0 (Hz) sang dải lag (mẫu) |
| `short_time_acf` | ACF baseline, trả về `R(k)` thô và `R(k)/R(0)` |
| `short_time_amdf` | AMDF baseline |
| `short_time_amdf_yin` | AMDF + cumulative mean normalization, chọn lag bằng "dip đầu tiên dưới ngưỡng", trả về `confidence = 1 - d'(lag)` |

**`train.py`**: duyệt `data/TinHieuHuanLuyen/`, gán nhãn từng khung theo `.lab` (theo tâm khung), tính `confidence`, in `meanV/stdV` (nhãn `v`) và `meanU/stdU` (nhãn `sil` + `uv` gộp chung), lưu mảng thô vào `results/` để đưa vào bước tìm ngưỡng (binary search / histogram từ BT1). Chạy: `python train.py`.

## Định dạng `.lab`

```
0.00    0.46    sil      # biên_trái  biên_phải  nhãn (giây)
0.46    1.39    v
1.39    1.50    uv
F0mean  122              # 2 dòng cuối: trung bình và độ lệch chuẩn F0 (Hz)
F0std   18
```

## Quy trình

1. `train.py`: thống kê `confidence` voiced/unvoiced trên tập huấn luyện
2. Tìm ngưỡng V/UV và tinh chỉnh `threshold` của YIN trên tập huấn luyện
3. `main.py`: chạy trên `TinHieuKiemThu/`, xuất figure, so F0mean/F0std và số lượng F0 với `.lab`
