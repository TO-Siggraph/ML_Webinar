#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'tensorflow_version 2.x')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import skimage


# In[ ]:


import tensorflow_datasets as tfds
merced_builder = tfds.builder('uc_merced')
# Download the dataset
merced_builder.download_and_prepare()


# In[ ]:


print(merced_builder.info)


# In[ ]:


f = merced_builder.info.features['image']
print(f)
PATCH_WIDTH = 128
PATCH_HEIGHT = 128


# In[ ]:


datasets = merced_builder.as_dataset()
full_ds = datasets['train']
fig = tfds.show_examples(merced_builder.info, full_ds)


# In[ ]:


# add gaussian noise to image
# note: this returns the noisy image as well as the original image
def add_noise(original, std_dev=0.1):
    noise = tf.random.normal(shape=tf.shape(original), mean=0.0, stddev=std_dev, dtype=tf.float32)
    noisy = tf.add(original, noise)
    noisy = tf.clip_by_value(noisy, 0.0, 1.0) # adding noise might make values > 1.0
    # return both the noisy and the normal image
    tensor_tuple = (noisy, original)
    return tensor_tuple


# In[ ]:


# extracts patches of given size from the image
def extract_patches(example, patch_width=PATCH_HEIGHT, patch_height=PATCH_HEIGHT):
  img = example['image']
  img = tf.image.convert_image_dtype(img, tf.float32)
  patches = tf.image.extract_patches([img], 
     sizes=[1, patch_height, patch_width, 1], 
     strides=[1, patch_height, patch_width, 1],
     rates=[1, 1, 1, 1],
     padding='SAME')
  img_tensor = tf.reshape(patches, [-1, patch_height, patch_width, 3])
  return tf.data.Dataset.from_tensor_slices(img_tensor)


# In[ ]:


256*256*3*32 +32


# In[ ]:


TRAIN_SIZE = int(2100*0.7)

train_ds = full_ds.take(TRAIN_SIZE)
test_ds = full_ds.skip(TRAIN_SIZE)


# In[ ]:


# prep dataset for training
train_ds = train_ds.flat_map(extract_patches)
train_ds = train_ds.map(add_noise)


# In[ ]:


# prep dataset for testing
# full-size patches in test data, want to test whole images
test_ds = test_ds.flat_map(lambda x: extract_patches(x, 256, 256))
test_ds = test_ds.map(add_noise)
test_ds = test_ds.batch(1) # one item per batch


# In[ ]:


for noisy, orig in train_ds.shuffle(100).take(5): #shuffling before 'take' will give us different images each time
    plt.figure()
    plt.imshow(noisy)
    plt.figure()
    plt.imshow(orig)


# In[ ]:


# repeat, shuffle and batch
train_ds = train_ds.repeat().shuffle(1024).batch(32)
# prefetch to asynchronously fetch batches while the model is training
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


def create_modelCNN():
    # build a sequential CNN model
    model = tf.keras.models.Sequential([
        # a stack of Conv2D layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',),
        tf.keras.layers.Conv2D(3, (3, 3), activation='relu', padding='same',),
    ])
    return model


# In[ ]:


def create_modelAE():
    # build a sequential AutoEncoder model (not used here: just for comparison of number of parameters)
    model = tf.keras.models.Sequential([
        # a stack of Conv2D layers
        tf.keras.layers.Dense(32, activation='relu', input_shape=(None, None, 3)),
        tf.keras.layers.Dense(128*128*3, activation='sigmoid'),
	      tf.keras.layers.Reshape((128, 128, 3))
	  ])
    return model


# In[ ]:


modelCNN = create_modelCNN()
print(modelCNN.summary())


# In[ ]:


modelAE = create_modelAE()
print(modelAE.summary())


# In[ ]:


model = modelCNN
# layers


# In[ ]:


# compile
model.compile(optimizer='adamax', loss='mae')


# In[ ]:


train_hist = model.fit(train_ds, epochs=10, steps_per_epoch=10)
                                 


# In[ ]:


NUM_PREDICT=3
prediction_batches = test_ds.take(NUM_PREDICT)


# In[ ]:


def show_results():
    # plot
    n = NUM_PREDICT  
    plt.figure(figsize=(15, 15))
    for i in range(n):
        # display noisy image
        ax = plt.subplot(4, n, i + 1)
        ax.set_title("Noisy")
        plt.imshow(noisy_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, i + 1 + n)
        ax.set_title("Denoised")
        plt.imshow(denoised_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display ground truth
        ax = plt.subplot(4, n, i + 1 + n + n)
        ax.set_title("Ground Truth")
        plt.imshow(hires_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display the difference (ssim)
        ax = plt.subplot(4, n, i + 1 + n + n + n)
        ax.set_title("Difference")
        plt.imshow(skimage.util.compare_images(hires_imgs[i], denoised_imgs[i], method='diff'))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# In[ ]:


hires_imgs = []
noisy_imgs = []
denoised_imgs = []
n=0
for nimg,himg in prediction_batches:
    # predict denoising from the noisy images
    pimg = model.predict(nimg)
    ssim = tf.image.ssim(tf.convert_to_tensor(np.expand_dims(pimg, 0)), himg, 1.0)
    print('SSIM: image:', n, ssim)
    # remove the extra batch dimension for matplotlib
    denoised_imgs.append(tf.squeeze(pimg))
    hires_imgs.append(tf.squeeze(himg))
    noisy_imgs.append(tf.squeeze(nimg))
    n += 1


# In[ ]:


show_results()

