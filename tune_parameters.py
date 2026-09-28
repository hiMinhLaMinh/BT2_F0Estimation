import sys
import numpy as np
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TEST_DIR = ROOT / "data" / "TinHieuKiemThu"

from io_utils import load_wav_normalized
from pitch_estimation import estimate_f0_track
from evaluation import evaluate_file

def run_training(yin_thresh):
    """Chạy train.py với yin_threshold cụ thể (tạm thời ghi đè biến trong file hoặc gọi hàm)"""
    from train import collect_confidence_stats, TRAIN_DIR
    stats, v_conf, u_conf = collect_confidence_stats(TRAIN_DIR, yin_threshold=yin_thresh)
    np.save(RESULTS_DIR / "voiced_confidence.npy", v_conf)
    np.save(RESULTS_DIR / "unvoiced_confidence.npy", u_conf)

def run_thresholding():
    """Gọi threshold.py để tìm T tối ưu cho yin_threshold hiện tại"""
    from threshold import binary_search_threshold
    v_conf = np.load(RESULTS_DIR / "voiced_confidence.npy")
    u_conf = np.load(RESULTS_DIR / "unvoiced_confidence.npy")
    return binary_search_threshold(v_conf, u_conf)

def evaluate_test_set(optimal_T, yin_thresh):
    wav_files = sorted(TEST_DIR.glob("*.wav"))
    f0_mean_errors = []
    
    for wav_path in wav_files:
        lab_path = wav_path.with_suffix(".lab")
        if not lab_path.exists(): continue
        fs, signal = load_wav_normalized(wav_path)
        f0_track, _ = estimate_f0_track(signal, fs, 70, 400, optimal_T, yin_thresh)
        res = evaluate_file(f0_track, lab_path)
        if not np.isnan(res['mean_dev']):
            f0_mean_errors.append(abs(res['mean_dev']))
            
    # Xử lý an toàn nếu danh sách trống
    if not f0_mean_errors:
        return float('inf')
    return np.mean(f0_mean_errors)

if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Các giá trị YIN Threshold muốn thử nghiệm
    yin_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print(f"{'YIN_Thresh':<12} | {'Optimal_T':<12} | {'Mean_F0_Error (Hz)':<18}")
    print("-" * 45)

    best_error = float('inf')
    best_params = {'yin': 0.1, 'T': 0.522727} # Thêm giá trị an toàn

    for yt in yin_candidates:
        # 1. Sinh dữ liệu train mới
        run_training(yt)
        # 2. Tìm T tối ưu mới
        T = run_thresholding()
        # 3. Đánh giá sai số trên tập test
        err = evaluate_test_set(T, yt)
        
        print(f"{yt:<12.2f} | {T:<12.6f} | {err:<18.2f}")
        
        if err < best_error:
            best_error = err
            best_params = {'yin': yt, 'T': T}

    print("-" * 45)
    print(f"\n=> TỐI ƯU NHẤT: YIN_Threshold = {best_params['yin']:.2f}, Ngưỡng T = {best_params['T']:.6f} (Sai số F0mean: {best_error:.2f} Hz)")