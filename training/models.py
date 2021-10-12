from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, \
    MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model
from keras import regularizers


def autoencoder_LSTM(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(48, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(12, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(12, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(48, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_CNN(X): 
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(256, 3, activation = 'relu', padding='same')(inputs)
    L2 = MaxPooling1D(2, padding='same')(L1)
    encoded = Conv1D(128, 2, activation='relu', padding='same')(L2)
    L3 = UpSampling1D(2)(encoded)
    L4 = Conv1D(256, 3, activation='relu', padding='same')(L3)
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L4)
    model = Model(inputs=inputs, outputs=output)
    return model
    