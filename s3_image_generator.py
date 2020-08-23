# Do the needed imports
import os
import sys
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.client import device_lib
from time import time
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import io
from more_itertools import chunked
from skimage.color import rgb2gray
import PIL
from tensorflow.python.client import device_lib
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))

# AWS S3 Access
session = boto3.session.Session()

s3 = session.client(
    service_name='s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=ecs_url
)

# Data Preprocessing
img_height, img_width = 300, 300
batch_size = 32
no_epochs = 100
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1
input_shape = (img_height, img_width, num_channels)
dataset_size = 1000


# Put image generator stuff here...
def s3_img_generator(s3, bucket, prefix=None, max_files=100, batch_size=32, color_mode='grayscale', class_mode=None):
    # Make sure the color mode is valid
    assert color_mode == "grayscale" or color_mode == "rgb", "ERROR: color mode must be 'grayscale' or 'rgb'"

    # Get the list of files out of S3 or Dell ECS
    #objects = s3.list_objects_v2(Bucket=BUCKET, Delimiter=',', Prefix='images-00111/00111', MaxKeys=max_files)['Contents']
    objects = s3.list_objects_v2(Bucket=BUCKET, Delimiter=',', Prefix=prefix, MaxKeys=max_files)['Contents']
    files = [x['Key'] for x in objects]

    # Break the data up in to batch sizes
    files = list(chunked(files, batch_size))

    # Iterate through the batches and yield the results
    while True:

        for batch in files:
            batch_img = []
            if class_mode != "input":
                batch_labels = [0] * len(batch)
            else:
                batch_labels = []
            for file in batch:
                obj = s3.get_object(Bucket=BUCKET, Key=file)
                #img = load_img(io.BytesIO(obj['Body'].read())) # The behavior has changed a bit in the latest tensorflow
                img = PIL.Image.open(io.BytesIO(obj['Body'].read()))
                imgarray = img_to_array(img)

                if color_mode == 'grayscale':
                    #print("SHAPE:", imgarray.shape)
                    #imgarray = np.reshape(imgarray, (300, 300, 1))
                    #imgarray = rgb2gray(imgarray)
                    imgarray = np.reshape(imgarray, (imgarray.shape[0], imgarray.shape[1], 1))

                batch_img += [imgarray]

                if class_mode == "input":
                    batch_labels += [imgarray]

            batch_img = np.array(batch_img)
            batch_labels = np.array(batch_labels)

            yield batch_img, batch_labels


train_generator = s3_img_generator(s3, BUCKET, prefix='images-00111/00111', max_files=dataset_size, batch_size=batch_size, class_mode='input')
val_generator = s3_img_generator(s3, BUCKET, prefix='images-00111/00111', max_files=dataset_size, batch_size=batch_size, class_mode='input')

#for i in list(train_generator):
#   print(i[0].shape, i[1].shape)
#
#print(20*"-")
#
#for i in list(val_generator):
#    print(i[0].shape, i[1].shape)