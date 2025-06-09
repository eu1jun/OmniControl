import numpy as np

# 1) Your existing AMASS_JOINT_MAP
AMASS_JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1,    'LKnee': 4,  'LAnkle': 7,  'LFoot': 10,
    'RHip': 2,    'RKnee': 5,  'RAnkle': 8,  'RFoot': 11,
    'LShoulder': 16, 'LElbow': 18, 'LWrist': 20,
    'RShoulder': 17, 'RElbow': 19, 'RWrist': 21,
    'spine1':3,   'spine2':6,   'spine3':9,
    'Neck':12,    'Head':15,
    'LCollar':13, 'Rcollar':14,
}

# 2) Invert it to get joint_names in index order
num_joints = max(AMASS_JOINT_MAP.values()) + 1
joint_names = [None] * num_joints
for name, idx in AMASS_JOINT_MAP.items():
    joint_names[idx] = name

# 3) Define the parent index for each joint (—1 for root)
parent_idxs = [-1] * num_joints
# hip root
parent_idxs[AMASS_JOINT_MAP['MidHip']] = -1
# legs
parent_idxs[AMASS_JOINT_MAP['LHip']]    = AMASS_JOINT_MAP['MidHip']
parent_idxs[AMASS_JOINT_MAP['LKnee']]  = AMASS_JOINT_MAP['LHip']
parent_idxs[AMASS_JOINT_MAP['LAnkle']] = AMASS_JOINT_MAP['LKnee']
parent_idxs[AMASS_JOINT_MAP['LFoot']]  = AMASS_JOINT_MAP['LAnkle']
parent_idxs[AMASS_JOINT_MAP['RHip']]    = AMASS_JOINT_MAP['MidHip']
parent_idxs[AMASS_JOINT_MAP['RKnee']]  = AMASS_JOINT_MAP['RHip']
parent_idxs[AMASS_JOINT_MAP['RAnkle']] = AMASS_JOINT_MAP['RKnee']
parent_idxs[AMASS_JOINT_MAP['RFoot']]  = AMASS_JOINT_MAP['RAnkle']
# spine → neck → head
parent_idxs[AMASS_JOINT_MAP['spine1']] = AMASS_JOINT_MAP['MidHip']
parent_idxs[AMASS_JOINT_MAP['spine2']] = AMASS_JOINT_MAP['spine1']
parent_idxs[AMASS_JOINT_MAP['spine3']] = AMASS_JOINT_MAP['spine2']
parent_idxs[AMASS_JOINT_MAP['Neck']]   = AMASS_JOINT_MAP['spine3']
parent_idxs[AMASS_JOINT_MAP['Head']]   = AMASS_JOINT_MAP['Neck']
# shoulders & arms off the upper spine
parent_idxs[AMASS_JOINT_MAP['LShoulder']] = AMASS_JOINT_MAP['spine3']
parent_idxs[AMASS_JOINT_MAP['LElbow']]    = AMASS_JOINT_MAP['LShoulder']
parent_idxs[AMASS_JOINT_MAP['LWrist']]    = AMASS_JOINT_MAP['LElbow']
parent_idxs[AMASS_JOINT_MAP['RShoulder']] = AMASS_JOINT_MAP['spine3']
parent_idxs[AMASS_JOINT_MAP['RElbow']]    = AMASS_JOINT_MAP['RShoulder']
parent_idxs[AMASS_JOINT_MAP['RWrist']]    = AMASS_JOINT_MAP['RElbow']
# collars off the neck
parent_idxs[AMASS_JOINT_MAP['LCollar']] = AMASS_JOINT_MAP['Neck']
parent_idxs[AMASS_JOINT_MAP['Rcollar']] = AMASS_JOINT_MAP['Neck']

# 4) Compute rest_offsets from the *first* frame of your motion
wrapper = np.load("results.npy", allow_pickle=True).item()
mot = wrapper["motion"][0]                  # shape (22,3,196)
frames = np.transpose(mot, (2,0,1))         # (196,22,3)
first_frame = frames[0]                     # (22,3)

rest_offsets = []
for i, p in enumerate(parent_idxs):
    if p < 0:
        rest_offsets.append((0.0, 0.0, 0.0))
    else:
        offset = first_frame[i] - first_frame[p]
        rest_offsets.append(tuple(offset.tolist()))

# Now you have exactly what you need:
print("joint_names =", joint_names)
print("parent_idxs =", parent_idxs)
print("rest_offsets =", rest_offsets)
