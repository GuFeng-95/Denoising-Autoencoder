import scipy.io as io
import numpy as np


for snr in [0.1,0.8,20,80]:
    pattern_data = io.loadmat('./pattern_data/Patterns.mat')['Q'].T
    g=(np.std(pattern_data)**2*snr)**0.5
    print(g)
    gauss = np.random.normal(0, g, pattern_data.shape).astype('float')
    pattern_data_noise = pattern_data + gauss

    np.save('./pattern_data/pattern_noise_{}.npy'.format(snr),pattern_data_noise)