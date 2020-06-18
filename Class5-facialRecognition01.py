#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'tensorflow_version 1.x')
from google.colab.patches import cv2_imshow
import cv2


# In[ ]:


# install the mtcnn package
get_ipython().system(u'pip install mtcnn')


# In[ ]:


import mtcnn
mtcnn_model = mtcnn.MTCNN()


# In[ ]:


# download an image
get_ipython().system(u'curl -o pixney_faces.jpeg https://i.pinimg.com/originals/cf/04/14/cf0414f971aa50c87140ed49223da33b.jpg')
disneypixarfaces = cv2.imread('pixney_faces.jpeg')
cv2_imshow(disneypixarfaces)


# In[ ]:


def recognize_faces(new_img):
  faces = mtcnn_model.detect_faces(new_img)
  for i in range(0, len(faces)):
      x, y, w, h = faces[i]['box']
      # draw rectangle
      cv2.rectangle(new_img, (x, y), (x+w, y+h), (255, 0, 0), 1)
  return new_img


# In[ ]:


img = disneypixarfaces
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
final_img = recognize_faces(img)
cv2_imshow(final_img[:,:,::-1])


# In[ ]:


# bonus: how to download and play video in colab
# Download sample video
get_ipython().system(u'curl -o sample.mp4 https://www.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4')
from IPython.display import HTML
from base64 import b64encode
mp4 = open('sample.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

