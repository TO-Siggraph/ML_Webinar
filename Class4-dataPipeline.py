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


# In[ ]:


# get the data as a tensorflow dataset
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


PATCH_WIDTH = 128
PATCH_HEIGHT = 128
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


print(test_ds)


# In[ ]:


some_patches = train_ds.take(5)
print(some_patches)


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

