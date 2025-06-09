import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore", "Optimal rotation")

# ---------------- 1. 데이터 로드 ----------------
wrapper = np.load("results.npy", allow_pickle=True).item()
mot     = wrapper["motion"][0]          # (J,3,F) = (22,3,196)
frames  = mot.transpose(2,0,1).copy()   # (F,J,3)
F, J    = frames.shape[:2]

# ---------------- 2. 스켈레톤 -------------------
parent = np.array([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
     9, 9,12,13,14,16,17,18,19       # ← 예: HumanML3D/AMASS 계층
])

# (i)  Rest offset : 첫 프레임을 레스트 포즈로 가정
rest_offset = np.zeros((J,3))
for j in range(J):
    p = parent[j]
    if p >= 0:
        rest_offset[j] = frames[0, j] - frames[0, p]

# (ii) 방향 벡터(정규화) – align_vectors 에 사용
rest_dir = np.zeros_like(rest_offset)
for j in range(1, J):
    v = rest_offset[j]
    rest_dir[j] = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([1,0,0])

# (iii) 자식 테이블
children = [[] for _ in range(J)]
for j, p in enumerate(parent):
    if p >= 0: children[p].append(j)

# ---------------- 3. BVH HIERARCHY -------------
def write_joint(fp, idx, indent):
    pad  = "  " * indent
    root = (parent[idx] < 0)
    fp.write(f"{pad}{'ROOT' if root else 'JOINT'} J{idx}\n{pad}{{\n")
    off = rest_offset[idx]
    fp.write(f"{pad}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
    ch = ("CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
          if root else
          "CHANNELS 3 Xrotation Yrotation Zrotation")
    fp.write(f"{pad}  {ch}\n")
    for c in children[idx]:
        write_joint(fp, c, indent+1)
    if not children[idx]:
        fp.write(f"{pad}  End Site\n{pad}  {{\n{pad}    OFFSET 0 0 0\n{pad}  }}\n")
    fp.write(f"{pad}}}\n")

# ---------------- 4. BVH 작성 -------------------
with open("omni_fk.bvh", "w") as fbvh:
    # -- 4-1) 계층
    fbvh.write("HIERARCHY\n")
    write_joint(fbvh, 0, 0)

    # -- 4-2) MOTION 헤더
    fbvh.write("MOTION\n")
    fbvh.write(f"Frames: {F}\nFrame Time: {1/30:.6f}\n")  # 30 fps 가정

    # -- 4-3) 각 프레임
    local_euler = np.zeros((F,J,3))

    for f in range(F):
        # 전역 회전 R_g, 전역 위치 P_g 를 FK 로 채워나감
        R_g = [None]*J
        P_g = [None]*J

        # --- (a) 루트 : 위치 = 실제 위치, 회전 = 두 벡터로 맞춤
        P_g[0] = frames[f,0]

        # spine1(3) & L-hip(1) 두 방향으로 root 회전을 고정
        ref  = [3, 1]
        Vcur = [(frames[f, c]-frames[f,0]) /
                np.linalg.norm(frames[f,c]-frames[f,0]) for c in ref]
        Vref = [rest_dir[c] for c in ref]
        R_g[0], _ = R.align_vectors(Vcur, Vref)   # rest→current

        # --- (b) BFS : pos_j = R_parent * offset_j + pos_parent
        order = [0] + [j for j in range(1,J)]     # 간단히 부모→자식 순
        for j in order[1:]:
            p = parent[j]

            # (1) 현재 global 위치는 Ground truth
            P_g[j] = frames[f, j]

            # (2) global 회전 R_j : 자식 방향 전체를 한꺼번에 정합
            if children[j]:                       # 내부 노드
                Vcur = []
                Vref = []
                for c in children[j]:
                    v = frames[f, c]-frames[f, j]
                    Vcur.append(v / np.linalg.norm(v))
                    Vref.append(rest_dir[c])
                R_g[j], _ = R.align_vectors(Vcur, Vref)
            else:                                # 말단 노드
                v = frames[f, j]-frames[f, p]
                R_g[j], _ = R.align_vectors([v/np.linalg.norm(v)],
                                             [rest_dir[j]])

            # --- local = R_parent⁻¹ · R_j  (BVH 에 기록되는 회전)
            local_euler[f, j] = (R_g[p].inv() * R_g[j]).as_euler('xyz', degrees=True)

        # 루트의 local 회전은 global 과 동일
        local_euler[f,0] = R_g[0].as_euler('xyz', degrees=True)

        # (c) BVH 채널 쓰기 (Root: 위치+회전, 나머지: 회전)
        line = [f"{P_g[0][0]:.6f}", f"{P_g[0][1]:.6f}", f"{P_g[0][2]:.6f}"] + \
               [f"{ang:.6f}" for ang in local_euler[f,0]]
        for j in range(1,J):
            line += [f"{ang:.6f}" for ang in local_euler[f,j]]
        fbvh.write(" ".join(line) + "\n")

print("✅  omni_fk.bvh 생성 완료")
