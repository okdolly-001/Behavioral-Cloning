from __future__ import unicode_literals
import argparse
import json
import utils

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, Conv2D, BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import *


kSEED = 5
SIDE_STEERING_CONSTANT = 0.25
NUM_BINS = 23


def batch_generator(images, angles, augment_data= True, batch_size=64):
    batch_images = []
    batch_angles = []
    sample_count = 0
    while True:
        for i in np.random.permutation(images.shape[0]):
            center_path = 'data/'+images.iloc[i,0]
            left_path = 'data/'+images.iloc[i,1]
            right_path = 'data/'+images.iloc[i,2]
            center_path = center_path.replace(" ", "")
            left_path = left_path.replace(" ", "")
            right_path = right_path.replace(" ", "")

            center_image = utils.load_image(center_path)
            angle = float(angles.iloc[i])
            batch_images.append(center_image)
            batch_angles.append(angle)

            sample_count += 1

            if augment_data:
                flipped_image = utils.flip(center_path)
                flipped_angle = -1.0 * angle
                batch_images.append(flipped_image)
                batch_angles.append(flipped_angle)

                tint_image = utils.tint_image(center_path)
                tint_angle = angle
                batch_images.append(tint_image)
                batch_angles.append(tint_angle)

                jittered_image, jitter_angle = utils.jitter_image(center_path,angle)
                batch_images.append(jittered_image)
                batch_angles.append(jitter_angle)

                left_image = utils.load_image(left_path)
                left_angle = min(1.0, angle+ SIDE_STEERING_CONSTANT)
                batch_images.append(left_image)
                batch_angles.append(left_angle)

                right_image = utils.load_image(right_path)
                right_angle = max(-1.0, angle - SIDE_STEERING_CONSTANT)
                batch_images.append(right_image)
                batch_angles.append(right_angle)

            if ((sample_count%batch_size == 0) or (sample_count % len(images)==0)):
                yield np.array(batch_images),np.array(batch_angles)
                batch_angles, batch_images = [], []

def create_model(lr=1e-3, activation='relu',nb_epoch=15):
    model = Sequential()
    # Lambda layer normalizes pixel values between 0 and 1
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    # Convolutional layer (1)
    model.add(Conv2D(24, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (2)
    model.add(Conv2D(36, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (3)
    model.add(Conv2D(48, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (4)
    model.add(Conv2D(64, (3,3), padding='same', activation=activation, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (5)
    model.add(Conv2D(64, (3,3), padding='same', activation=activation, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Flatten Layer
    model.add(Flatten())
    # Dense Layer (1)
    model.add(Dense(1164, activation=activation))
    # Dense layer (2)
    model.add(Dense(100, activation=activation))
    # Dense layer (3)
    model.add(Dense(50, activation=activation))
    # Dense layer (4)
    model.add(Dense(10, activation=activation))
    # Dense layer (5)
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer=Adam(lr=lr, decay=lr / nb_epoch), loss='mse')
    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", help="Initial learning rate",
                           type=float, default=1e-3, required=False)
    argparser.add_argument("--nb_epoch", help="Number of epochs to train for",
                           type=int, default=15, required=False)
    argparser.add_argument("--activation", help="Activation function to use",
                           type=str, default='relu', required=False)
    args = argparser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models/')


    
    print('[INFO] Loading Data.')
    file_name = 'data/driving_log.csv'
    images,angles = utils.load_data(file_name)
    X_train, X_val, y_train, y_val = train_test_split(images,angles,test_size=0.15,random_state = kSEED)

    print('[INFO] Preprocessing images and augmenting data')
    train_generator = batch_generator(X_train,y_train,augment_data=True)
    test_generator = batch_generator(X_val, y_val,augment_data=False)

    print('[INFO] Creating Model.')
    model = create_model(args.lr, args.activation, args.nb_epoch)
    checkpoint = ModelCheckpoint('models/model-{epoch:03d}.h5',monitor='val_loss', verbose=0, save_best_only= True, mode = 'auto')
    print('[INFO] Training model')
    model.fit_generator(train_generator, steps_per_epoch= len(X_train)/64,epochs=args.nb_epoch,validation_data=test_generator,
                        callbacks=[checkpoint],validation_steps=len(X_val),verbose=1)
    print('[INFO] Saving Model')
    model.save_weights('models/model.h5',True)
    with open('models/model.json', 'w') as outfile:
        json.dump(model.to_json,outfile)
