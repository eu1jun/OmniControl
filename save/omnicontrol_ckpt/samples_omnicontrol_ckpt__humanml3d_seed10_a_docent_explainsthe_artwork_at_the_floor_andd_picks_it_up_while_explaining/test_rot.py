import numpy as np

# ----- 1) 데이터 로드 -------------------------------------------------
wrapper     = np.load("results_rot.npy", allow_pickle=True).item()
print(wrapper.keys())
print(wrapper['motion'].shape)
if False:
    mot         = wrapper["motion"][0]               # (J,3,F) 혹은 (22,3,196)
    frames      = np.transpose(mot, (2,0,1))         # (F, J, 3)
    F, J        = frames.shape[:2]