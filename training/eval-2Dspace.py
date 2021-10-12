import numpy as np 
from pycbc import frame
from gwpy.timeseries import TimeSeries
import os
import argparse
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gwpy.timeseries import TimeSeries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, \
    MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras import regularizers
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError
import tensorflow as tf
from keras.models import load_model


def reshape_data(X_transformed, timesteps): 
    #reshape data into model-friendly input
    X_train = X_transformed[:]
    print(X_train.shape)
    if X_train.shape[0] % timesteps == 0:
        print('reshaping')
        X_train = X_train.reshape((-1, timesteps, 1))
    else:
        print('reshaping and cutting off extra')
        X_train = X_train[:-1 * int(X_train.shape[0] % timesteps)].reshape((-1, timesteps, 1))
    print(X_train.shape)
    X_train = tf.constant(X_train)
    return X_train

def loss_train(outdir, model_type, detector, timefirst, timesecond):
    model = load_model('%s/best_model.hdf5'%(outdir))

    if model_type == 'LSTM':
        timesteps = 128
    elif model_type == 'CNN':
        timesteps = 1024
    elif model_type == 'transformer':
        timesteps = 128
    
    X = h5.File('../data/preprocessed_strain_IP.h5', 'r')[detector + '_strain'][1024*409*timefirst:1024*409*timesecond].reshape((-1, 1))
    X = np.clip(X, -3, 3)
    X = MinMaxScaler().fit_transform(X)
    X = reshape_data(X, timesteps)
    loss_fn = MeanSquaredError(reduction='none')
    
    X_output = model.predict(X, verbose=1)
    losses_train = loss_fn(X_output, X)
    del X_output, X
    
    losses_train = np.mean(losses_train, axis=1).reshape(-1)[:]
    return losses_train
'''    
outdir_H1 = 'output_H1_CNN_minmaxclip_IP'
outdir_L1 = 'output_L1_CNN_minmaxclip_IP'

model_L1 = load_model('%s/best_model.hdf5'%(outdir_L1))
model_H1 = load_model('%s/best_model.hdf5'%(outdir_H1))

#detector = 'H1'
model_type = 'CNN'

if model_type == 'LSTM':
    timesteps = 128
elif model_type == 'CNN':
    timesteps = 1024
elif model_type == 'transformer':
    timesteps = 128

print('loading data...')
X_L1 = h5.File('../data/preprocessed_strain_IP.h5', 'r')['L1_strain'][:].reshape((-1, 1))
X_H1 = h5.File('../data/preprocessed_strain_IP.h5', 'r')['H1_strain'][:].reshape((-1, 1))
X_L1 = np.clip(X_L1, -3, 3)
X_H1 = np.clip(X_H1, -3, 3)
print('scaling data...')
X_L1 = MinMaxScaler().fit_transform(X_L1)
X_H1 = MinMaxScaler().fit_transform(X_H1)

X_L1 = reshape_data(X_L1)
X_H1 = reshape_data(X_H1)

loss_fn = MeanSquaredError(reduction='none')

# Measure losses for train and test set
X_train_output_L1 = model_L1.predict(X_L1, verbose=1)
X_train_output_H1 = model_L1.predict(X_H1, verbose=1)

losses_train_L1 = loss_fn(X_train_output_L1, X_L1)
losses_train_H1 = loss_fn(X_train_output_H1, X_H1)

del X_train_output_H1, X_train_output_L1, X_H1, X_L1

losses_train_L1 = np.mean(losses_train_L1, axis=1).reshape(-1)[:]
losses_train_H1 = np.mean(losses_train_H1, axis=1).reshape(-1)[:]
'''

outdir_H1 = 'output_H1_LSTM_minmaxclip_IP'
outdir_L1 = 'output_L1_LSTM_minmaxclip_IP_v2'
model = 'LSTM'
model_type = model

losses_train_L1 = np.concatenate((loss_train(outdir_L1, model, 'L1', 0, 100), loss_train(outdir_L1, model, 'L1', 100, 200), loss_train(outdir_L1, model, 'L1', 200, 300), loss_train(outdir_L1, model, 'L1' , 300, 400)))
losses_train_H1 = np.concatenate((loss_train(outdir_H1, model, 'H1', 0, 100), loss_train(outdir_H1, model, 'H1', 100, 200), loss_train(outdir_H1, model, 'H1', 200, 300), loss_train(outdir_H1, model, 'H1', 300, 400)))

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(losses_train_L1, losses_train_H1, s=2)
plt.title(model_type + ' 2D Model loss - full month')
plt.ylabel('loss H1')
plt.xlabel('loss L1')
#plt.legend(['train set', 'test set'], loc='upper left')
plt.ylim([0.0, 0.01])
plt.xlim([0.0, 0.01])
plt.savefig(f'{outdir_L1}/loss_2d_distribution_zoomed.jpg')


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(losses_train_L1, losses_train_H1, s=2)
plt.title(model_type + ' 2D Model loss - full month')
plt.ylabel('loss H1')
plt.xlabel('loss L1')
#plt.legend(['train set', 'test set'], loc='upper left')
plt.ylim([0.0, 0.03])
plt.xlim([0.0, 0.03])
plt.savefig(f'{outdir_L1}/loss_2d_distribution.jpg')


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


polar_r, polar_phi = cart2pol(losses_train_L1, losses_train_H1)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
pi = np.pi
plt.xticks(np.arange(0, 2*pi/3, step=(pi/6)), ['0', 'π/6','π/3','π/2'])
plt.scatter(polar_phi, polar_r, s=2)
plt.title(model_type + ' 2D Model loss - full month')
plt.ylabel('r')
plt.xlabel('phi')
#plt.legend(['train set', 'test set'], loc='upper left')
#plt.ylim([0.0, 0.02])
#plt.xlim([0.0, 0.02])
plt.savefig(f'{outdir_L1}/loss_2d_distribution_polar.jpg')





