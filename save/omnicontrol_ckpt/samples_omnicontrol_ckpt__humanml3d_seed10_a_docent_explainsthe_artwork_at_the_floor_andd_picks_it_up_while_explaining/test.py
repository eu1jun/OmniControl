import numpy as np
from scipy.spatial.transform import Rotation as R

# ----- 1) 데이터 로드 -------------------------------------------------
wrapper     = np.load("results.npy", allow_pickle=True).item()
mot         = wrapper["motion"][0]               # (J,3,F) 혹은 (22,3,196)
frames      = np.transpose(mot, (2,0,1))         # (F, J, 3)
F, J        = frames.shape[:2]

# ----- 2) 스켈레톤 정의 ----------------------------------------------
parent      = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    12,12,12,9, 9,16,17,18,19
])
rest_offset = np.array([
 [ 0.00000,  0.00000,  0.00000],
 [ 0.05738, -0.08563, -0.00393],[-0.06148,-0.09064,-0.01039],
 [-0.00525,  0.12064, -0.05191],
 [ 0.04635, -0.35125,  0.16625],[ 0.00753,-0.38406,-0.04554],
 [-0.00895,  0.13800,  0.05042],
 [-0.03808, -0.38263, -0.19705],[ 0.06815,-0.40716,-0.08737],
 [-0.00169,  0.05471,  0.01615],
 [ 0.08374, -0.06380,  0.09198],[-0.05000,-0.04336,  0.13614],
 [-0.01347,  0.21495,  0.03411],
 [ 0.07752, -0.09255, -0.02677],[-0.07458,-0.10422,-0.01360],
 [-0.04716,  0.06857,  0.06012],
 [ 0.19359,  0.12506, -0.00370],[-0.20435, 0.08273,  0.01003],
 [ 0.02009, -0.21512, -0.13960],[-0.02505,-0.25398,-0.07035],
 [ 0.07178, -0.25580, -0.00092],[-0.03964,-0.22658,  0.13868]
])

# 자식 리스트 생성
children = [[] for _ in range(J)]
for j, p in enumerate(parent):
    if p >= 0:
        children[p].append(j)

# rest bone 방향(단순 offset)과 정규화
rest_dir = np.array([v / np.linalg.norm(v) for v in rest_offset])

# 로컬 회전 저장용 배열 (root 제외)
local_euler = np.zeros((F, J, 3), dtype=float)

# ----- 3) 프레임별 회전 계산 -----------------------------------------
for f in range(F):
    # 1) 글로벌 회전 초기화
    global_rot = [R.identity() for _ in range(J)]

    # 2) root 회전: hip→첫번째 자식 간 방향 정합
    first_child = children[0][0]     # 보통 0의 첫 자식이 spine
    u_rest      = rest_dir[first_child]
    u_curr      = frames[f, first_child] - frames[f, 0]
    u_rest_n    = u_rest / np.linalg.norm(u_rest)
    u_curr_n    = u_curr / np.linalg.norm(u_curr)
    R0, _       = R.align_vectors([u_curr_n], [u_rest_n])
    root_euler  = R0.as_euler('XYZ', degrees=True)
    global_rot[0] = R0

    # 3) 나머지 본들에 대해
    for j in range(1, J):
        p = parent[j]

        # rest & current 벡터
        v_rest = rest_dir[j]
        v_curr = frames[f, j] - frames[f, p]

        # 정규화
        v_rest_n = v_rest / np.linalg.norm(v_rest)
        v_curr_n = v_curr / np.linalg.norm(v_curr)

        # 정합 회전 (rest → current)
        Rg, _ = R.align_vectors([v_curr_n], [v_rest_n])

        # 로컬 회전 분리: parent^-1 * global
        local_rot = global_rot[p].inv() * Rg

        # Euler 추출 (intrinsic XYZ)
        local_euler[f, j] = local_rot.as_euler('XYZ', degrees=True)

        # 글로벌 회전 누적
        global_rot[j] = global_rot[p] * local_rot

    # 4) MOTION 라인 작성
    if f == 0:
        # 파일 열기
        fbvh = open("omni_fixed.bvh", "w")
        # HIERARCHY 헤더
        fbvh.write("HIERARCHY\n")
        def write_joint(idx, indent):
            pad  = '  ' * indent
            name = f"J{idx}"
            chan = ("CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
                    if parent[idx] < 0 else
                    "CHANNELS 3 Xrotation Yrotation Zrotation")
            fbvh.write(f"{pad}{'ROOT' if parent[idx]<0 else 'JOINT'} {name}\n")
            fbvh.write(f"{pad}{{\n")
            off = rest_offset[idx]
            fbvh.write(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
            fbvh.write(f"{pad}  {chan}\n")
            for c in children[idx]:
                write_joint(c, indent+1)
            if parent[idx] >= 0:
                fbvh.write(f"{pad}  End Site\n")
                fbvh.write(f"{pad}  {{\n")
                fbvh.write(f"{pad}    OFFSET 0.000000 0.000000 0.000000\n")
                fbvh.write(f"{pad}  }}\n")
            fbvh.write(f"{pad}}}\n")
        write_joint(0, 0)
        # MOTION 헤더
        fbvh.write(f"MOTION\nFrames: {F}\nFrame Time: {1/30:.6f}\n")

    # root position + rotation
    xyz = frames[f, 0]
    line = [
        f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}",
        f"{root_euler[0]:.6f}", f"{root_euler[1]:.6f}", f"{root_euler[2]:.6f}"
    ]
    # 나머지 본들 Euler
    for j in range(1, J):
        e = local_euler[f, j]
        line += [f"{e[0]:.6f}", f"{e[1]:.6f}", f"{e[2]:.6f}"]
    fbvh.write(" ".join(line) + "\n")

# 파일 닫기
fbvh.close()
print("omni_fixed.bvh 생성 완료!")
