import scipy.io as io
import numpy as np


for g in [0.35,1,5,10]:
    cylinder_data = np.load('./cylinder_data/cylinder_clean.npy')
    gauss = np.random.normal(0, g, cylinder_data.shape).astype('float')
    cylinder_data_noise = cylinder_data + gauss
    np.save('./cylinder_data/cylinder_noise_{}.npy'.format(g),cylinder_data_noise)