import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import PIL
from PIL import Image
from scipy.misc import imresize, toimage
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Dropout, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import SGD
from keras.constraints import maxnorm

def convert_data(df, path='data/'):
    for item in df[['center', 'steering']].iterrows():
        try:
            img = Image.open(path + item[1].center)
            x = np.array(img) / 255. - 0.5
            x = imresize(x, [100, 200])[30:-10, :]
            np.save('data_npy/' + item[1].center + '.npy', x)
        except:
            print('Cannot load ', item[1].center)


def get_data(df):
    X = []
    y = []
    for item in df[['center', 'steering']].iterrows():
        X.append(np.load('data_npy/' + item[1].center + '.npy'))
        y.append(item[1].steering)
    y = np.array(y)
    X = np.array(X)
    return X, y


def data_generator(df, batch_size=32):
    df.reset_index(inplace=True, drop=True)
    n_iter = math.ceil(df.shape[0] / batch_size)
    while 1:
        for i in range(n_iter)[::-1]:
            start = i * batch_size
            end = start + batch_size - 1
            yield get_data(df.loc[start:end])


def conv_model(dropout=0.4, epoch=4, postfix=False, model=None):
    if model == None:
        model = Sequential()
        model.add(Conv2D(24, 5, 5, input_shape=(60, 200, 3)))
        model.add(MaxPooling2D((2, 2), border_mode='same'))
        model.add((Dropout(dropout)))
        model.add(Activation('relu'))

        model.add(Conv2D(36, 5, 5))
        model.add(MaxPooling2D((2, 2)))
        model.add((Dropout(dropout)))
        model.add(Activation('relu'))

        model.add(Conv2D(48, 5, 5))
        model.add(MaxPooling2D((2, 2)))
        model.add((Dropout(dropout)))
        model.add(Activation('relu'))

        model.add(Conv2D(64, 3, 3))
        model.add(MaxPooling2D((1, 1)))
        model.add((Dropout(dropout)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(1000, init='normal',
            activation='relu', W_constraint=maxnorm(3)))
        model.add(Dense(100, init='normal',
            activation='relu', W_constraint=maxnorm(3)))
        model.add(Dense(20, init='normal', activation='relu',
            W_constraint=maxnorm(3)))
        model.add(Dense(1, init='normal'))

    model.summary()

    model.compile(loss='mean_squared_error',
            optimizer='adam')
    history = model.fit_generator(data_generator(train), samples_per_epoch=train.shape[0],
            nb_epoch=epoch, validation_data=Xy_valid,
            verbose=2)
    if postfix == False:
        fname = 'model'
    else:
        fname = 'model_rl_drop_%.1f_epoch_%d' % (dropout, last_epoch)
    model.save_weights('%s.h5' % fname)
    model_json = model.to_json()
    with open("%s.json" % fname, "w") as json_file:
        json_file.write(model_json)
    return model, history

if __name__ == '__main__':

    # Loading the driving log data
    # df_mid: car centered
    df_mid = pd.read_csv('data/driving_log.csv')
    # df_edges: car centered
    df_edges = pd.read_csv('edges/driving_log.csv', names=df_mid.columns)

    #
    df_edges_left = df_edges[df_edges.steering > 0]
    df_edges_right = df_edges[df_edges.steering < 0]
    #
    df_edges_left.steering = 0.2
    df_edges_right.steering = -0.2

    # Remove paths to camara images
    df_edges.loc[:, 'center'] = df_edges.apply(
            lambda x: x.center[25:], axis=1)
    df_edges_right.loc[:, 'center'] = df_edges_right.apply(
            lambda x: x.right[26:], axis=1)
    df_edges_left.loc[:, 'center'] = df_edges_left.apply(
            lambda x: x.left[26:], axis=1)

    # Preprocess the data
    convert_data(df_mid, 'data/')
    convert_data(df_edges, 'edges/')
    convert_data(df_edges_left, 'edges/')
    convert_data(df_edges_right, 'edges/')

    # Concateneting driving log dataframes
    df = pd.concat([df_mid, df_edges, df_edges_left, df_edges_right])[
            ['center', 'steering']]
    print('Mean steering angle: ' df.steering.mean())
    print('Std steering angle: ' df.steering.std())

    # Split data into train, test and validation sets
    train_valid, test = train_test_split(df, test_size=0.1)
    train, valid = train_test_split(train_valid, test_size=0.1)
    print('Train set shape: ', train.shape)
    print('Validation set shape: ', valid.shape)
    print('Test set shape: ', test.shape)

    # Load validation data
    Xy_valid = get_data(valid)
    # Train the network
    model, history = conv_model(0.4, epoch=6, postfix=True)

    # Load test data
    Xy_test = get_data(test)
    # Test accuracy
    mse = model.evaluate(Xy_test[0], Xy_test[1])
    print('Test MSE: %.3f' % mse)
