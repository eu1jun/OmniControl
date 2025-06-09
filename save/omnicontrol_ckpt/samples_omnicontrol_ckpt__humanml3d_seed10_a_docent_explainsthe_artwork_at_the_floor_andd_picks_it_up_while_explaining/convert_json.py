import numpy as np, json

wrapper     = np.load("results.npy", allow_pickle=True).item()
mot         = wrapper["motion"][0]             # (J,3,F)
frames      = np.transpose(mot, (2,0,1))       # (F, J, 3)

# 각 프레임을 {"data": [...flattened floats...]} 형태의 객체로 감싸기
data = {
    "frames": [
        {"data": frame.reshape(-1).tolist()}
        for frame in frames
    ]
}

with open("motion.json", "w") as f:
    json.dump(data, f)
