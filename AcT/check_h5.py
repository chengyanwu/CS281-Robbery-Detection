import h5py

f = h5py.File('AcT_pretrained_weights/AcT_base_1_0.h5', 'r')

print(f)
print(list(f.keys()))