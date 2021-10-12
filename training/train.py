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
from models import autoencoder_LSTM, autoencoder_CNN

detector = 'L1'
model_type = 'LSTM'

if model_type == 'LSTM':
    timesteps = 128
elif model_type == 'CNN':
    timesteps = 1024
elif model_type == 'transformer':
    timesteps = 128

print('loading data...')
X = h5.File('../data/preprocessed_strain_IP.h5', 'r')[detector + '_strain'][:].reshape(-1)
print('scaling data...')
X = np.clip(X, -3, 3).reshape((-1, 1))
X_transformed = MinMaxScaler().fit_transform(X)

#reshape data into model-friendly input
X_train = X_transformed[:]
print(X_train.shape)
if X_train.shape[0] % timesteps == 0:
    print('reshaping')
    X_train = X_train.reshape((-1, timesteps, 1))
else:
    print('reshaping and cutting off extra')
    X_train = X_train[:-1 * int(X_train.shape[0] % timesteps)].reshape((-1, timesteps, 1))
shape = X_train.shape
print(shape)
X_train = tf.constant(X_train)

'''
X_test = X_transformed[:]

del X_transformed 

if X_test.shape[0] % timesteps == 0:
    print('reshaping')
    X_test = X_test.reshape((-1, timesteps, 1))
else:
    print('reshaping and cutting off extra')
    X_test = X_test[:-1 * int(X_test.shape[0] % timesteps)].reshape((-1, timesteps, 1))
shape = X_test.shape
print(shape)
X_test = tf.constant(X_test)
'''

# Define the model
if model_type == 'LSTM':
    model = autoencoder_LSTM(X_train)
elif model_type == 'CNN': 
    model = autoencoder_CNN(X_train)
elif model_type == 'transformer': 
    model = autoencoder_transformer(X_train)
    
model.compile(optimizer='adam', loss='mse')
model.summary()
outdir = 'output_' + detector + '_' + model_type + '_minmaxclip_IP_v2'

# Fit the model to the data
nb_epochs = 150
batch_size = 2048
early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=0, mode='min')
mcp_save = ModelCheckpoint(f'{outdir}/best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size, validation_split=0.2,
                    callbacks=[early_stop, mcp_save])
model.save(f'{outdir}/last_model.hdf5')

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'{outdir}/loss_training.jpg')

loss_fn = MeanSquaredError(reduction='none')

# Measure losses for train and test set
X_train_output = model.predict(X_train, verbose=1)
loss_train = loss_fn(X_train_output, X_train)
np.save(outdir + '/train_output_' + model_type + '_' + detector + '.npy', np.mean(loss_train, axis=1).reshape(-1)[:])

'''
#X_test_output = model.predict(X_test, verbose=1)

losses_test = loss_fn(X_test_output, X_test)
losses_train = loss_fn(X_train_output, X_train)

averaged_losses_test = np.mean(losses_test, axis=1).reshape(-1)
averaged_losses_train = np.mean(losses_train, axis=1).reshape(-1)

import matplotlib.pyplot as plt
plt.plot([*range(len(averaged_losses_train))], averaged_losses_train)
plt.plot(np.sum(([*range(len(averaged_losses_test))], 
                 [len(averaged_losses_train) for i in range(len(averaged_losses_train))]), 
                axis=0), averaged_losses_test)

plt.title(model_type + ' model loss train/test set')
plt.ylabel('loss')
plt.xlabel('timestep')
plt.legend(['train set', 'test set'], loc='upper left')
plt.savefig(f'{outdir}/loss_timestep.jpg')
'''