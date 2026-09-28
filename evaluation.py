"""
Đánh giá định lượng kết quả ước lượng F0 so với file .lab tương ứng.

Theo đề BT2: tính F0mean và F0std của các giá trị F0 thuật toán tìm được,
rồi tính độ lệch của hai giá trị này so với F0mean/F0std trong file .lab.
Ngoài ra đếm số F0 tìm được (số khung voiced) để đối chiếu khoảng
minNumF0..maxNumF0 mà giảng viên nhắc trên lớp.

Quy ước đầu vào: f0_track là mảng F0 (Hz) theo từng khung; khung unvoiced
đặt là np.nan (hoặc 0) -> sẽ bị loại khi thống kê.
"""

from pathlib import Path

import numpy as np

from io_utils import parse_lab_file


def f0_statistics(f0_track):
    """Trả về (F0mean, F0std, số F0 hợp lệ). F0std là độ lệch chuẩn tổng thể (ddof=0)."""
    f0 = np.asarray(f0_track, dtype=np.float64)
    f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if len(f0) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(f0)), float(np.std(f0)), len(f0)


def evaluate_f0(f0_track, f0mean_gt, f0std_gt, min_num_f0=None, max_num_f0=None):
    """
    So F0mean/F0std của thuật toán với giá trị chuẩn.

    Độ lệch = ước lượng - chuẩn (có dấu, dương = ước lượng cao hơn chuẩn),
    kèm phần trăm so với giá trị chuẩn.
    min_num_f0 / max_num_f0 (tuỳ chọn): khoảng số F0 hợp lý để gắn nhãn thấp/cao.
    """
    mean_est, std_est, n = f0_statistics(f0_track)

    result = {
        "f0mean_est": mean_est,
        "f0std_est": std_est,
        "f0mean_gt": f0mean_gt,
        "f0std_gt": f0std_gt,
        "mean_dev": mean_est - f0mean_gt,
        "mean_dev_pct": 100.0 * (mean_est - f0mean_gt) / f0mean_gt,
        "std_dev": std_est - f0std_gt,
        "std_dev_pct": 100.0 * (std_est - f0std_gt) / f0std_gt if f0std_gt else float("nan"),
        "num_f0": n,
        "num_f0_status": None,
    }

    if min_num_f0 is not None and n < min_num_f0:
        result["num_f0_status"] = "thấp (thuật toán quá 'lười')"
    elif max_num_f0 is not None and n > max_num_f0:
        result["num_f0_status"] = "cao (thuật toán quá rườm rà)"
    elif min_num_f0 is not None or max_num_f0 is not None:
        result["num_f0_status"] = "trong khoảng"

    return result


def evaluate_file(f0_track, lab_path, min_num_f0=None, max_num_f0=None):
    """Đọc F0mean/F0std chuẩn từ file .lab rồi đánh giá f0_track."""
    _, f0mean_gt, f0std_gt = parse_lab_file(lab_path)
    if f0mean_gt is None or f0std_gt is None:
        raise ValueError(f"Không tìm thấy F0mean/F0std trong {lab_path}")

    result = evaluate_f0(f0_track, f0mean_gt, f0std_gt, min_num_f0, max_num_f0)
    result["file"] = Path(lab_path).stem
    return result


def format_report(results):
    """Bảng tổng hợp cho nhiều file (dùng để in ra màn hình / đưa vào báo cáo)."""
    header = (
        f"{'File':<14}{'F0mean':>9}{'(chuẩn)':>9}{'lệch':>8}{'lệch%':>8}"
        f"{'F0std':>9}{'(chuẩn)':>9}{'lệch':>8}{'lệch%':>8}{'#F0':>6}"
    )
    lines = [header, "-" * len(header)]

    for r in results:
        lines.append(
            f"{r['file']:<14}{r['f0mean_est']:>9.1f}{r['f0mean_gt']:>9.1f}"
            f"{r['mean_dev']:>+8.1f}{r['mean_dev_pct']:>+8.1f}"
            f"{r['f0std_est']:>9.1f}{r['f0std_gt']:>9.1f}"
            f"{r['std_dev']:>+8.1f}{r['std_dev_pct']:>+8.1f}{r['num_f0']:>6d}"
        )
        if r["num_f0_status"] and r["num_f0_status"] != "trong khoảng":
            lines.append(f"    ! số F0 {r['num_f0_status']}")

    n_failed = sum(1 for r in results if r["num_f0"] == 0)
    mean_abs = np.nanmean([abs(r["mean_dev"]) for r in results]) if n_failed < len(results) else float("nan")
    std_abs = np.nanmean([abs(r["std_dev"]) for r in results]) if n_failed < len(results) else float("nan")
    lines.append("-" * len(header))
    lines.append(f"Trung bình |lệch| trên {len(results) - n_failed}/{len(results)} file: F0mean {mean_abs:.2f} Hz, F0std {std_abs:.2f} Hz")
    if n_failed:
        lines.append(f"    ! {n_failed} file không tìm được F0 nào, đã loại khỏi trung bình")
    return "\n".join(lines)
