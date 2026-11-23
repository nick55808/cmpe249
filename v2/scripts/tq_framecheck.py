import sys; sys.path.insert(0, "/workspace/teleopq")
import h5py, numpy as np
from tq_annotate import pose
np.set_printoptions(precision=3, suppress=True)
f = h5py.File("/workspace/isaaclab/datasets/dataset_annotated_gr1.hdf5", "r"); d = f["data/demo_0"]
a = d["actions"][:]; pa = d["processed_actions"][:]
for side, sl in (("left", slice(0, 7)), ("right", slice(7, 14))):
    ref = d["obs/datagen_info/target_eef_pose/" + side][:]
    for name, src_a in (("actions", a), ("processed", pa)):
        T = pose(src_a[:, sl][:, :3], src_a[:, sl][:, 3:7])
        print(side, name, "pos err max", np.abs(T[:, :3, 3] - ref[:, :3, 3]).max(), "rot err max", np.abs(T[:, :3, :3] - ref[:, :3, :3]).max())
    eef = d["obs/datagen_info/eef_pose/" + side][:]
    print(side, "ref target[0]\n", ref[0], "\naction[0]", a[0, sl], "\neef[0]\n", eef[0])
    # try shift
    for s in (-1, 1):
        T = pose(a[:, sl][:, :3], a[:, sl][:, 3:7])
        if s == 1: e = np.abs(T[1:] - ref[:-1]).max()
        else: e = np.abs(T[:-1] - ref[1:]).max()
        print(side, "shift", s, "max err", e)
