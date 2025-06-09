import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore", "Optimal rotation is not uniquely or poorly defined")

# ----- 1) 데이터 로드 -------------------------------------------------
wrapper = np.load("results.npy", allow_pickle=True).item()
mot     = wrapper["motion"][0]                 # (J,3,F)  or  (22,3,196)
frames  = mot.transpose(2, 0, 1).copy()        # (F, J, 3)
F, J    = frames.shape[:2]

# ----- 2) 스켈레톤 정의 ----------------------------------------------
parent      = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    9, 9, 12, 13, 14, 16, 17, 18, 19
])

rest_offset = np.zeros((J,3))
for j in range(J):
    p = parent[j]
    if p < 0: continue
    rest_offset[j] = (frames[:,j] - frames[:,p]).mean(axis=0)

# 자식 리스트 생성
children = [[] for _ in range(J)]
for j, p in enumerate(parent):
    if p >= 0:
        children[p].append(j)

rest_dir = np.zeros_like(rest_offset)
for j in range(1,J):
    norm = np.linalg.norm(rest_offset[j])
    if norm > 1e-8:
        rest_dir[j] = rest_offset[j]/norm
    else:
        rest_dir[j] = np.array([1,0,0])

# ----------- 3) BVH HIERARCHY ----------------------------------------
def write_joint(fp, idx, indent):
    pad  = "  " * indent
    name = f"J{idx}"
    root = parent[idx] < 0
    channels = ("CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
                if root else
                "CHANNELS 3 Xrotation Yrotation Zrotation")
    fp.write(f"{pad}{'ROOT' if root else 'JOINT'} {name}\n")
    fp.write(f"{pad}{{\n")
    off = rest_offset[idx]
    fp.write(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
    fp.write(f"{pad}  {channels}\n")
    for c in children[idx]:
        write_joint(fp, c, indent + 1)
    if not children[idx]:                      # leaf only
        fp.write(f"{pad}  End Site\n{pad}  {{\n")
        fp.write(f"{pad}    OFFSET 0.000000 0.000000 0.000000\n{pad}  }}\n")
    fp.write(f"{pad}}}\n")

with open("omni_fixed.bvh", "w") as fbvh:
    # ----------- HIERARCHY --------------
    fbvh.write("HIERARCHY\n")
    write_joint(fbvh, 0, 0)

    # ----------- MOTION header ----------
    fbvh.write("MOTION\n")
    fbvh.write(f"Frames: {F}\n")
    fbvh.write(f"Frame Time: {1/30:.6f}\n")

    # ----------- 4) per-frame -----------
    local_euler = np.zeros((F, J, 3), dtype=float)

    for f in range(F):
        global_rot = [R.identity() for _ in range(J)]

        # ----- ROOT 회전 (MidHip) ---------------------------------
        #   p0 = MidHip(0), p1 = spine1(3), p2 = LHip(1)
        A_rest = np.stack([
            rest_dir[3],   # MidHip→spine1
            rest_dir[1]    # MidHip→LHip
        ], axis=0)                         # shape (2,3)

        A_curr = np.stack([
            frames[f, 3] - frames[f, 0],
            frames[f, 1] - frames[f, 0]
        ], axis=0)
        A_curr /= np.linalg.norm(A_curr, axis=1, keepdims=True)

        R0, _ = R.align_vectors(A_rest, A_curr)  # rest → current
        global_rot[0] = R0
        root_euler = R0.as_euler('xyz', degrees=True)

        # ------ 나머지 본들 ---------------
        for j in range(1, J):
            p = parent[j]

            v_rest = rest_dir[j]
            v_curr = frames[f, j] - frames[f, p]
            v_curr /= np.linalg.norm(v_curr)

            Rg, _ = R.align_vectors([v_rest], [v_curr])   # rest → current
            local_rot = global_rot[p].inv() * Rg
            local_euler[f, j] = local_rot.as_euler('xyz', degrees=True)
            global_rot[j] = global_rot[p] * local_rot

        # ------ 채널 쓰기 -----------------
        line = [
            f"{frames[f,0,0]:.6f}", f"{frames[f,0,1]:.6f}", f"{frames[f,0,2]:.6f}",
            f"{root_euler[0]:.6f}", f"{root_euler[1]:.6f}", f"{root_euler[2]:.6f}"
        ]
        for j in range(1, J):
            ex, ey, ez = local_euler[f, j]
            line += [f"{ex:.6f}", f"{ey:.6f}", f"{ez:.6f}"]
        fbvh.write(" ".join(line) + "\n")

print("✅  omni_fixed.bvh 생성 완료")
# 3) subset joint 정의
use_joints     = [0,1,2,3]
idx_map        = {orig:i for i, orig in enumerate(use_joints)}
parent_small   = np.array([ idx_map[parent[j]] if parent[j] in idx_map else -1
                            for j in use_joints ])
rest_offset_sm = rest_offset[use_joints]           # (4,3)
frames_sm      = frames[:, use_joints, :]          # (F,4,3)

# 4) 자식 리스트, rest_dir 계산
children_sm = [[] for _ in range(4)]
for j,p in enumerate(parent_small):
    if p>=0: children_sm[p].append(j)

rest_dir_sm = np.zeros_like(rest_offset_sm)
for j in range(1,4):
    rest_dir_sm[j] = rest_offset_sm[j] / np.linalg.norm(rest_offset_sm[j])

# 5) BVH 생성
with open("subset_4joints.bvh","w") as fbvh:
    # HIERARCHY
    fbvh.write("HIERARCHY\n")
    def write_joint(idx, indent):
        pad  = "  "*indent
        is_root = (parent_small[idx]<0)
        name = f"J{use_joints[idx]}"
        chan = ("CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
                if is_root else
                "CHANNELS 3 Xrotation Yrotation Zrotation")
        fbvh.write(f"{pad}{'ROOT' if is_root else 'JOINT'} {name}\n")
        fbvh.write(f"{pad}{{\n")
        off = rest_offset_sm[idx]
        fbvh.write(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
        fbvh.write(f"{pad}  {chan}\n")
        for c in children_sm[idx]:
            write_joint(c, indent+1)
        if not children_sm[idx]:
            fbvh.write(f"{pad}  End Site\n{pad}  {{\n")
            fbvh.write(f"{pad}    OFFSET 0.000000 0.000000 0.000000\n{pad}  }}\n")
        fbvh.write(f"{pad}}}\n")
    write_joint(0,0)

    # MOTION 헤더
    fbvh.write("MOTION\n")
    fbvh.write(f"Frames: {F}\n")
    fbvh.write(f"Frame Time: {1/30:.6f}\n")

    # 프레임별 채널 값 계산
    local_euler = np.zeros((F,4,3), float)
    for f in range(F):
        global_rot = [R.identity() for _ in range(4)]

        # 1) 루트 회전 (MidHip 기준 spine1(3), LHip(1))
        A_rest = np.stack([rest_dir_sm[3], rest_dir_sm[1]], 0)
        A_curr = np.stack([
            frames_sm[f,3] - frames_sm[f,0],
            frames_sm[f,1] - frames_sm[f,0]
        ], 0)
        A_curr /= np.linalg.norm(A_curr, axis=1, keepdims=True)
        R0, _ = R.align_vectors(A_rest, A_curr)
        global_rot[0] = R0
        root_euler = R0.as_euler('xyz', degrees=True)

        # 2) J1, J2, J3 local 회전
        for j in range(1,4):
            p = parent_small[j]
            v_rest = rest_dir_sm[j]
            v_curr = frames_sm[f,j] - frames_sm[f,p]
            v_curr /= np.linalg.norm(v_curr)
            Rg, _ = R.align_vectors([v_rest], [v_curr])    # rest → current
            local_rot = global_rot[p].inv() * Rg
            local_euler[f,j] = local_rot.as_euler('xyz', degrees=True)
            global_rot[j] = global_rot[p] * local_rot

        # 3) 채널 라인 작성
        line = [
            f"{frames_sm[f,0,0]:.6f}", f"{frames_sm[f,0,1]:.6f}", f"{frames_sm[f,0,2]:.6f}",
            f"{root_euler[0]:.6f}", f"{root_euler[1]:.6f}", f"{root_euler[2]:.6f}"
        ]
        for j in range(1,4):
            ex, ey, ez = local_euler[f,j]
            line += [f"{ex:.6f}", f"{ey:.6f}", f"{ez:.6f}"]
        fbvh.write(" ".join(line) + "\n")

print("✅ subset_4joints.bvh 생성 완료")