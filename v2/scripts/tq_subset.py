"""Copy selected demos of an HDF5 dataset into a new file (renumbered demo_0..). Usage: tq_subset.py IN OUT demo_3 [demo_7 ...]"""
import sys, h5py
src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
fi = h5py.File(src, "r"); fo = h5py.File(dst, "w"); go = fo.create_group("data")
for k, v in fi["data"].attrs.items(): go.attrs[k] = v
total = 0
for i, n in enumerate(names):
    fi["data"].copy(fi["data"][n], go, name=f"demo_{i}"); total += int(fi["data"][n].attrs.get("num_samples", len(fi["data"][n]["actions"])))
go.attrs["total"] = total
fo.close(); print("wrote", dst, names, "total samples", total)
