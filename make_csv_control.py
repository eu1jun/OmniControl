import numpy as np, csv, argparse, os, math
from pathlib import Path

def load_csv(path):
    pts = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            pts.append([float(row['x']), float(row['y']), float(row['z'])])
    return np.asarray(pts, np.float32)          # (N,3)

def resample(arr, tgt_len):
    """linear 1-D time resampling"""
    if len(arr) == tgt_len: return arr
    idx = np.linspace(0, len(arr)-1, tgt_len)
    out = np.empty((tgt_len, 3), np.float32)
    for j in range(3):
        out[:, j] = np.interp(idx, np.arange(len(arr)), arr[:, j])
    return out

def make_control(csv_path, motion_len=196, pelvis_joint=0, pelvis_height=1.0,
                 mean_path='dataset/humanml_spatial_norm/Mean_raw.npy',
                 std_path='dataset/humanml_spatial_norm/Std_raw.npy',
                 out_path='control.npy'):
    pts = load_csv(csv_path)                 # (N,3)
    pts = resample(pts, motion_len)          # (T,3)
    # ─── X/Z swap ───
    # pts[:,0] ← old Z, pts[:,2] ← old X
    pts = pts[:, [2, 1, 0]]                  # now each row is [Z, Y, X]
    
    # 이미 CSV에서 높이를 올려둬서 주석
    # pts[:,1] += pelvis_height              
    
    pts /= 2.0
    pts[:,1] /= 2.0

    mean = np.load(mean_path)
    std  = np.load(std_path)
    ctrl = np.zeros((motion_len, 22, 3), np.float32)   # T×J×3
    mask = np.ones_like(pts[:,0])                      # dense
    normed = (pts - mean[pelvis_joint]) / std[pelvis_joint]
    ctrl[:, pelvis_joint, :] = normed * mask[:,None]
    np.save(out_path, {'pos':ctrl.reshape(motion_len,-1),
                       'sigma':mask[:,None].astype(np.float32)})
    print(f"[make_csv_control] saved control to {out_path}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--motion_len', type=int, default=196)
    p.add_argument('--pelvis_height', type=float, default=1.0)
    p.add_argument('--out', default='control.npy')
    args = p.parse_args()
    make_control(args.csv, args.motion_len, out_path=args.out,
                 pelvis_height=args.pelvis_height)
