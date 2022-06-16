#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[1]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# In[2]:


get_ipython().system('pip install Pillow==8.4')


# In[3]:


import io
import datetime
import pandas as pd
from PIL import Image
import requests
import io
import glob, os, sys, time, uuid

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw

from video_indexer import VideoIndexer
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import TrainingStatusType
from msrest.authentication import CognitiveServicesCredentials


# In[4]:


# Todo: add resources
CONFIG = {
    'SUBSCRIPTION_KEY': '686e8611f5584bda8817fef515aaf42a',
    'LOCATION': 'trial',
    'ACCOUNT_ID': 'e27ddf3d-59b1-49c8-b8fc-0848d377528c'
}

video_analysis = VideoIndexer(
    vi_subscription_key=CONFIG['SUBSCRIPTION_KEY'],
    vi_location=CONFIG['LOCATION'],
    vi_account_id=CONFIG['ACCOUNT_ID']
)


# In[5]:


video_analysis.check_access_token()


# ### TODO: Upload 30-sec video from local source

# In[6]:


uploaded_video_id = video_analysis.upload_to_video_indexer(
    input_filename = 'boarding-pass-video-gilles-kepnang.mp4', 
    video_name = 'kepnang-video-boarding', 
    video_language = 'English'
)


# In[12]:


uploaded_video_id


# In[13]:


info = video_analysis.get_video_info(uploaded_video_id, video_language='English')


# In[14]:


info


# In[15]:


#Face thumbnail extracted
if len(info['videos'][0]['insights']['faces'][0]['thumbnails']):
    print("We found {} faces in this video.".format(str(len(info['videos'][0]['insights']['faces'][0]['thumbnails']))))


# In[16]:


info['videos'][0]['insights']['faces'][0]['thumbnails']


# In[26]:


# Todo: get Thumbnail ID from the Analysis JSON 
images = []
img_raw = []
img_strs = []
video_id = 'dc33d76c1b'
for each_thumb in info['videos'][0]['insights']['faces'][0]['thumbnails']:
    if 'fileName' in each_thumb and 'id' in each_thumb:
        file_name = each_thumb['fileName']
        thumb_id = each_thumb['id']
        # Todo: get the image code for the thumbnail ID 
        # using the video_analysis.get_thumbnail_from_video_indexer object
        img_code = video_analysis.get_thumbnail_from_video_indexer(video_id, thumb_id)
        img_strs.append(img_code)
        img_stream = io.BytesIO(img_code)
        img_raw.append(img_stream)
        img = Image.open(img_stream)
        images.append(img)


# In[27]:


for img in images:
    print(img.info)
    plt.figure()
    plt.imshow(img)


# In[28]:


# Save the images
i = 1
for img in images:
    print(type(img))
    img.save('video-analyzer-face' + str(i) + '.jpg')
    i= i+ 1


# In[29]:


# Verify the download process 
get_ipython().system('ls video-analyzer-face*.jpg')


# # Creating Person Model Based on Faces in the Video
# 
# Note: this section is the same as the Facial recognition Demo. You can jump to the next section "Additional Resource: uploading a video from local disk to Video Analyzer portal" if you don't want to perform personal model again.
# 
# ### We have already downloaded and saved the face thumbnails in the previous steps
# - We will be using those face thumbnails here to build the Person model

# In[17]:


PERSON_GROUP_ID = str(uuid.uuid4())
person_group_name = 'person-gilles'

# Note if this UUID already used earlier, you will get an error 


# In[69]:


## This code is taken from Azure Face SDK 
## ---------------------------------------
def build_person_group(client, person_group_id, pgp_name):
    print('Create and build a person group...')
    # Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
    print('Person group ID:', person_group_id)
    client.person_group.create(person_group_id = person_group_id, name=person_group_id)

    # Create a person group person.
    human_person = client.person_group_person.create(person_group_id, pgp_name)
    # Find all jpeg human images in working directory.
    human_face_images = [file for file in glob.glob('*.jpg') if file.startswith("human-face")]
    # Add images to a Person object
    for image_p in human_face_images:
        with open(image_p, 'rb') as w:
            client.person_group_person.add_face_from_stream(person_group_id, human_person.person_id, w)

    # Train the person group, after a Person object with many images were added to it.
    client.person_group.train(person_group_id)

    # Wait for training to finish.
    while (True):
        training_status = client.person_group.get_training_status(person_group_id)
        print("Training status: {}.".format(training_status.status))
        if (training_status.status is TrainingStatusType.succeeded):
            break
        elif (training_status.status is TrainingStatusType.failed):
            client.person_group.delete(person_group_id=PERSON_GROUP_ID)
            sys.exit('Training the person group has failed.')
        time.sleep(5)


# In[70]:


GILLES_FACE_KEY = "f058b7cb938f4223995ee25f69b20727"
GILLES_FACE_ENDPOINT = "https://kepnangface.cognitiveservices.azure.com/"


# In[71]:


# Create a client
face_client = FaceClient(GILLES_FACE_ENDPOINT, CognitiveServicesCredentials(GILLES_FACE_KEY))


# In[72]:


face_client.api_version


# In[73]:


build_person_group(face_client, PERSON_GROUP_ID, person_group_name)


# In[74]:


'''
Detect all faces in query image list, then add their face IDs to a new list.
'''
def detect_faces(client, query_images_list):
    print('Detecting faces in query images list...')

    face_ids = {} # Keep track of the image ID and the related image in a dictionary
    for image_name in query_images_list:
        image = open(image_name, 'rb') # BufferedReader
        print("Opening image: ", image.name)
        time.sleep(5)

        # Detect the faces in the query images list one at a time, returns list[DetectedFace]
        faces = client.face.detect_with_stream(image)  

        # Add all detected face IDs to a list
        for face in faces:
            print('Face ID', face.face_id, 'found in image', os.path.splitext(image.name)[0]+'.jpg')
            # Add the ID to a dictionary with image name as a key.
            # This assumes there is only one face per image (since you can't have duplicate keys)
            face_ids[image.name] = face.face_id

    return face_ids


# In[75]:


get_ipython().system('ls video-analyzer-face*.jpg')


# In[76]:


my_face_images = [file for file in glob.glob('*.jpg') if file.startswith("video-analyzer-face")]
print(my_face_images)


# In[77]:


for img in my_face_images:
    with open(img, 'rb') as img_code:
        img_view_ready = Image.open(img_code)
        plt.figure()
        plt.imshow(img_view_ready)


# In[78]:


ids = detect_faces(face_client, my_face_images)


# In[79]:


ids


# In[80]:


### TODO: Matching Face From The Person Model With Face From Video Analyzer 


# In[81]:


# Todo: choose a image from the Video Analyzer and change the local path to read the image
dl_image = open('/home/workspace/CCP Photos LLC-1776 (2).jpg', 'rb')

# Detect faces in that image
dl_faces = face_client.face.detect_with_stream(dl_image)


# In[82]:


# View Face ID and then save it into the list of already saved Face IDs
for face in dl_faces:
    print('Face ID', face.face_id, 'found in image', dl_image)
    # Add the ID to a dictionary with image name as a key.
    # This assumes there is only one face per image (since you can't have duplicate keys)
    ids['video-analyzer.png'] = face.face_id


# In[83]:


# Now, you should have n + 1 Face IDs in the list
ids


# In[84]:


# Todo: enter the video analyzer face ID of from the output of the cell above
get_the_face_id_from_the_video_analyzer = '21bbf12c-eb3a-473c-b020-3d0de18710cb'


# In[85]:


# Identify the face from the video analyzer
person_gp_results = face_client.face.identify([get_the_face_id_from_the_video_analyzer], PERSON_GROUP_ID)


# In[86]:


for result in person_gp_results:
    if result.candidates:
        for candidate in result.candidates:
            print("The Identity match confidence is {}".format(candidate.confidence))
    else:
        print("Can't verify the identity with the person group")


# In[87]:


info['summarizedInsights']['emotions']


# In[ ]:




