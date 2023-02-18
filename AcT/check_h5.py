import h5py

f = h5py.File('AcT_pretrained_weights/AcT_base_1_0.h5', 'r')

# print(f)
# print(list(f.keys()))

with h5py.File('AcT_pretrained_weights/AcT_micro_1_0.h5', 'r') as f:
    for k in f.keys():
        for l in f[k].keys():
            for m in f[k][l].keys():
                print(k+ " : " + m + " : " + str(f[k][l][m].shape))