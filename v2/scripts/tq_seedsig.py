import h5py, numpy as np
f = h5py.File("/workspace/isaaclab/datasets/dataset_annotated_gr1.hdf5", "r")
for k in sorted(f["data"].keys()):
    d = f["data"][k]
    sig = d["obs/datagen_info/subtask_term_signals/idle_right"][:].reshape(-1)
    a = d["actions"][:]
    first = int(np.argmax(sig)) if sig.any() else -1
    rpos = a[:, 7:10]; lpos = a[:, 0:3]
    rspeed = np.linalg.norm(np.diff(rpos, axis=0), axis=1); lspeed = np.linalg.norm(np.diff(lpos, axis=0), axis=1)
    rmove = int(np.argmax(rspeed > 0.002)); lmove = int(np.argmax(lspeed > 0.002))
    rh = a[:, 25:]; lh = a[:, 14:25]
    rgrip = np.abs(np.diff(rh, axis=0)).sum(1); lgrip = np.abs(np.diff(lh, axis=0)).sum(1)
    print(k, "len", len(a), "idle_right first True", first, "n True", int(sig.sum()), "right moves from", rmove, "left moves from", lmove,
          "right-hand finger change steps", np.nonzero(rgrip > 0.05)[0][:6].tolist(), "left finger change", np.nonzero(lgrip > 0.05)[0][:6].tolist())
print("datagen_info keys:", list(f["data/demo_0/obs/datagen_info"].keys()))
print("env_args mimic cfg?", [x for x in f["data"].attrs.keys()])
