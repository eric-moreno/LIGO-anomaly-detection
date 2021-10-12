import os
import numpy as np 
from pycbc import frame
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from utils import as_stride, get_PSD, pearson_shift
import h5py
import logging

# set up logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

RATE = 4096
INPUT_DUR = 1
NPERSEG = int(max(2 * RATE, 2048))
INPUT_SIZE = int(INPUT_DUR * RATE)
STEP = 50
BATCH_SIZE = 256
PEARSON_SHIFT = 40
cutoff_sec = 10


def condition(x): return x < 126*2

def RemoveBadDQ(strain_array, DQ_array_L1, DQ_array_H1): 
    
    DQ_array = np.sum((DQ_array_L1, DQ_array_H1), axis=0)
    output = [idx for idx, element in enumerate(DQ_array) if condition(element)]
    output_4096 = []
    for i in output: 
        output_4096.append(np.arange(int(i*4096), int(i*4096)+4096))
    return np.delete(strain_array, np.array(output_4096).flatten().astype(int)), len(output_4096)/4096
    
def filters(array, sample_frequency, resample_frequency, DQ_array_L1, DQ_array_H1):
    """ Apply preprocessing such as whitening and bandpass """
    
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    H1_duty_cycle = (1 - np.isnan(strain).sum() / len(strain)) 
    if np.any(np.isnan(strain)):
        logger.warning('NaN found in H1 strain, replacing with median value')
        strain = np.nan_to_num(strain, strain.median())
    
    freqs, H1_Pxx = get_PSD('psd_H1', strain, nperseg=NPERSEG, duty_cycle=H1_duty_cycle, plot=True, plot_dir="Output")
    H1_ASD = FrequencySeries(np.sqrt(H1_Pxx), df=RATE / NPERSEG)
    strain1k_whiten = strain.whiten(asd=H1_ASD)
    bandpass = strain1k_whiten.bandpass(30, 350) 
    strain_removedbadDQ, frac = RemoveBadDQ(bandpass, DQ_array_L1, DQ_array_H1)
    resampled_strain = TimeSeries(np.array(strain_removedbadDQ).reshape(-1), sample_rate=int(sample_frequency))
    strain = resampled_strain.resample(resample_frequency)[1024*cutoff_sec:-1024*cutoff_sec]
    print(np.max(strain))
    
    return strain.reshape(-1)

data_names_H1 = sorted(os.listdir('../data/strain_4k/H1/'))
data_names_L1 = sorted(os.listdir('../data/strain_4k/L1/'))

with h5py.File('../data/preprocessed_strain_IP.h5', "a") as f:
    dset_H1 = f.create_dataset('H1_strain', (0,), maxshape=(None,),
                            dtype='f4')   
    dset_L1 = f.create_dataset('L1_strain', (0,), maxshape=(None,),
                            dtype='f4')   

    for i, j in zip(data_names_H1, data_names_L1): 
        print(i, j)
        strain_H1 = frame.read_frame('../data/strain_4k/H1/'+ i, 'H1:GWOSC-4KHZ_R1_STRAIN')
        strain_L1 = frame.read_frame('../data/strain_4k/L1/' + j, 'L1:GWOSC-4KHZ_R1_STRAIN')
        DQ_H1 = frame.read_frame('../data/strain_4k/H1/'+ i, 'H1:GWOSC-4KHZ_R1_DQMASK')
        DQ_L1 = frame.read_frame('../data/strain_4k/L1/'+ j, 'L1:GWOSC-4KHZ_R1_DQMASK')
        X_H1 = filters(strain_H1, 4096, 1024, np.array(DQ_L1), np.array(DQ_H1))
        X_L1 = filters(strain_L1, 4096, 1024, np.array(DQ_L1), np.array(DQ_H1))

        dset_H1.resize(dset_H1.shape[0]+X_H1.shape[0], axis=0)
        dset_L1.resize(dset_L1.shape[0]+X_L1.shape[0], axis=0)
        print(dset_H1.shape, dset_L1.shape)
        dset_H1[-X_H1.shape[0]:] = X_H1
        dset_L1[-X_L1.shape[0]:] = X_L1
    
    
    
    
