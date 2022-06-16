#!/usr/bin/env python
# coding: utf-8

# ### Please install the required Python modules/SDKs

# In[31]:


get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# # Azure Custom Vision - Image Classification Demo

# ## Importing utility functions and Python modules

# In[32]:


import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import os, time, uuid


# In[33]:


from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials


# ### Resources:
# - Azure Custom Vision Endpoint
# - Training Reource ID and Key
# - Prediction Resource ID and Key

# ## Make sure you have the correct Training and Prediction Endpoints, Keys and Resource IDs separately

# In[34]:


TRAINING_ENDPOINT = "https://kepnangcustomviz.cognitiveservices.azure.com/"
training_key = "dc967e3e4c8e49ed9d885f0abe51cdbf"
training_resource_id = "/subscriptions/0cad54ed-eb76-40ec-a04f-c9d4854fd2f6/resourceGroups/Kepnang-RG-001/providers/Microsoft.CognitiveServices/accounts/kepnangcustomviz"


# In[35]:


PREDICTION_ENDPOINT = "https://kepnangcustomviz-prediction.cognitiveservices.azure.com/"
prediction_key = "b83e7a74d5d343e08900b4355e2b1621"
prediction_resource_id = "/subscriptions/0cad54ed-eb76-40ec-a04f-c9d4854fd2f6/resourceGroups/Kepnang-RG-001/providers/Microsoft.CognitiveServices/accounts/kepnangcustomviz-Prediction"


# ## Instantiate and authenticate the training client with endpoint and key 

# In[36]:


training_credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)


# In[37]:


trainer.api_version


# ## Creating Training Project First

# In[38]:


# Create a new project
print ("Training project created. Proceed to the next cell.")
project_name = "project1step04-python-sdk"
project = trainer.create_project(project_name)


# ## Getting Project Details as collective information 

# In[39]:


project.as_dict()


# ## Adding Tags based on training requirements
# - We have 4 tags in the training process 
#   - glass
#   - key
#   - lighter
#   - smartphone

# In[40]:


glass_tag = trainer.create_tag(project.id, "Glass")


# In[41]:


key_tag = trainer.create_tag(project.id, "Key")


# In[42]:


lighter_tag = trainer.create_tag(project.id, "Lighter")


# In[43]:


smartphone_tag = trainer.create_tag(project.id, "Smartphone")


# ## Upload Traning Data 

# ### Enter local file system location of the traning images 
# - All training images are saved in the file system within this workspace environment. 
# - You can click on the "Jupyter" icon on the top left of this workspace to view these image folders:
#     - You will find all glass images in the `glass-images` folder
#     - You will find all key images in the `key-images` folder
#     - You will find all the lighter images in the `lighter-images` folder
#     - You will find all the smartphone images in the `smartphone-images` folder
#     - There are also three test images for you to perform predictions later.

# In[44]:


# Get current working directory
# The output will give you the "local_image_path" used in the cell below
get_ipython().system('pwd')


# In[45]:


local_image_path = '/home/workspace/'


# In[46]:


# Some code is taken from Azure SDK Sample and added my own code here
def upload_images_for_training(local_project_id, local_img_folder_name, image_tag_id):
    image_list = []
    files = os.listdir(os.path.join (local_image_path, local_img_folder_name))
    for file in files:
        full_path = os.path.join(local_image_path, local_img_folder_name, file)
        if os.path.isfile(full_path) and (full_path.endswith('.jpg') or full_path.endswith('.JPG') or full_path.endswith('.jpeg')):
            with open(os.path.join (local_image_path, local_img_folder_name, file), "rb") as image_contents:
                image_list.append(ImageFileCreateEntry(name=file, contents=image_contents.read(), tag_ids=[image_tag_id]))
                
    upload_result = trainer.create_images_from_files(local_project_id, ImageFileCreateBatch(images=image_list))
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)
        exit(-1)
    return upload_result


# In[47]:


glass_upload_result = upload_images_for_training(project.id, 'glass-images', glass_tag.id)


# In[48]:


glass_upload_result.is_batch_successful


# In[49]:


key_upload_result = upload_images_for_training(project.id, 'key-images', key_tag.id)


# In[50]:


key_upload_result.is_batch_successful


# In[51]:


lighter_upload_result = upload_images_for_training(project.id, 'lighter-images', lighter_tag.id)


# In[52]:


lighter_upload_result.is_batch_successful


# In[53]:


smartphone_upload_result = upload_images_for_training(project.id, 'smartphone-images', smartphone_tag.id)


# In[54]:


smartphone_upload_result.is_batch_successful


# ## Start the Image Classification Training
# - We will keep checking the training progress every 10 seconds

# In[55]:


iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    print ("Waiting 10 seconds...")
    time.sleep(10)


# ## After training is complete, let's look at the Model Performance

# In[56]:


iteration.as_dict()


# In[57]:


iteration_list = trainer.get_iterations(project.id)
for iteration_item in iteration_list:
    print(iteration_item)


# In[58]:


model_perf = trainer.get_iteration_performance(project.id, iteration_list[0].id)


# In[59]:


model_perf.as_dict()


# ## Publishing the Model to the Project Endpoint

# In[60]:


# Setting the Iteration Name, this will be used when Model training is completed
# Please choose a name favorable to you.
publish_iteration_name = "sdk-iteration1"


# In[61]:


# The iteration is now trained. Publish it to the project endpoint
trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
print ("Done!")


# ## Instantiate and authenticate the prediction client with endpoint and key

# In[62]:


prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)


# In[63]:


predictor.api_version


# ## Performing Prediction
# - Using the predictor object 

# In[64]:


def perform_prediction(image_file_name):
    with open(os.path.join (local_image_path,  image_file_name), "rb") as image_contents:
        results = predictor.classify_image(project.id, publish_iteration_name, image_contents.read())
        # Display the results.
        for prediction in results.predictions:
            print("\t" + prediction.tag_name +
                  ": {0:.2f}%".format(prediction.probability * 100))


# ## The test images are stored in the local file system of this workspace
# * You will perform prediction twice below

# In[65]:


# To list the folders/files in your current working directory
# The name of any image file can be used as "file_name" in the cell below
get_ipython().system('ls')


# In[66]:


file_name = 'lighter_test_set_1of5.jpg'


# In[67]:


# Pick one test image file name from the output of the previous cell
# Use the same image file name for this cell and the next one
perform_prediction(file_name)


# In[68]:


# Checking the image
with open(os.path.join (local_image_path, file_name), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# In[69]:


file_name_2 = 'lighter_test_set_2of5.jpg'


# In[70]:


# Perform prediction again using another image
perform_prediction(file_name_2)


# In[71]:


# Checking the second image
with open(os.path.join (local_image_path, file_name_2), 'rb') as img_code:
    img_view_ready = Image.open(img_code)
    plt.figure()
    plt.imshow(img_view_ready)


# In[ ]:




