import h5py, numpy as np
np.set_printoptions(precision=3, suppress=True)
f = h5py.File("/workspace/isaaclab/datasets/dataset_annotated_gr1.hdf5", "r")
print("== seeds: object height and right-hand pose around the subtask boundary")
for k in sorted(f["data"].keys()):
    d = f["data"][k]
    sig = d["obs/datagen_info/subtask_term_signals/idle_right"][:].reshape(-1); b = int(np.argmax(sig))
    op = d["obs/datagen_info/object_pose/object"][:]; st = d["states/rigid_object/object/root_pose"][:]
    z = op[:, 2, 3]; z0 = z[0]
    rh = d["actions"][:, 25:]
    lift = np.nonzero(z - z0 > 0.01)[0]; lift1 = int(lift[0]) if len(lift) else -1
    close = np.linalg.norm(rh - rh[0], axis=1)
    print(k, "boundary", b, "z0 %.3f" % z0, "dz@b %.3f" % (z[b]-z0), "first lift>1cm", lift1,
          "dz@b-10 %.3f dz@b+10 %.3f" % (z[max(b-10,0)]-z0, z[min(b+10,len(z)-1)]-z0),
          "hand-closure@b %.2f max %.2f" % (close[b], close.max()),
          "| datagen obj pos", op[0,:3,3], "state root", st[0,:3])
g = h5py.File("/workspace/isaaclab/datasets/v2/lat0.hdf5", "r")
d = g["data"][sorted(g["data"].keys())[0]]
print("== v2 lat0 demo keys:", list(d["obs"].keys()))
print("obs object_pos[0]", d["obs/object_pos"][0], "state root[0]", d["states/rigid_object/object/root_pose"][0,:3], "robot root", d["states/articulation/robot/root_pose"][0,:3], "obs robot_root_pos", d["obs/robot_root_pos"][0])
print("obs right_eef_pos[0]", d["obs/right_eef_pos"][0], "action right pos[0]", d["actions"][0,7:10])
z = d["obs/object_pos"][:,2]; print("v2 demo len", len(z), "first lift>1cm", int(np.argmax(z - z[0] > 0.01)))
