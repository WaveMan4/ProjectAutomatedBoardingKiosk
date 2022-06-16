#!/usr/bin/env python
# coding: utf-8

# In[63]:


## Utility Functions
get_ipython().system(' activate ai-azure-c1')

import sys

sys.path.append("/opt/conda/envs/ai-azure-c1/lib/python3.8/site-packages")


# In[98]:


import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
from io import BytesIO


# In[104]:


def show_image_in_cell(img_url): 
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    plt.show()


# In[105]:


## Setup


# In[106]:


import os
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.formrecognizer import FormTrainingClient
from azure.core.credentials import AzureKeyCredential


# In[68]:


AZURE_FORM_RECOGNIZER_ENDPOINT = "https://project1formrecognizerdigid.cognitiveservices.azure.com/"
AZURE_FORM_RECOGNIZER_KEY = "c4b3929528d243e3ad9def1befc8195e"


# In[69]:


endpoint = AZURE_FORM_RECOGNIZER_ENDPOINT
key = AZURE_FORM_RECOGNIZER_KEY


# In[70]:


form_training_client = FormTrainingClient(endpoint=endpoint, credential=AzureKeyCredential(key))


# In[71]:


saved_model_list = form_training_client.list_custom_models()


# In[72]:


trainingDataUrl = "https://project1step02.blob.core.windows.net/boarding?sp=rcwdl&st=2022-06-02T02:52:01Z&se=2022-06-06T10:52:01Z&skoid=5f0cb329-aa4f-4571-8f0a-8eb5af801702&sktid=2b5b2a7c-d342-454f-b5d3-69ca89275ad2&skt=2022-06-02T02:52:01Z&ske=2022-06-06T10:52:01Z&sks=b&skv=2020-08-04&spr=https&sv=2020-08-04&sr=c&sig=VrMXrd7ieA86%2FHPEKh2Myuc2yuYx%2F7gHg%2FUFtCASIYc%3D"


# In[73]:


training_process = form_training_client.begin_training(trainingDataUrl, use_training_labels=False)
custom_model = training_process.result()


# In[74]:


custom_model


# In[75]:


custom_model.model_id


# In[76]:


custom_model.status


# In[77]:


custom_model.training_started_on


# In[78]:


custom_model.training_completed_on


# In[79]:


custom_model.training_documents


# In[80]:


for doc in custom_model.training_documents: 
    print("Document name: {}".format(doc.name))
    print("Document status: {}".format(doc.status))
    print("Document page count: {}".format(doc.page_count))
    print("Document errors: {}".format(doc.errors))


# In[81]:


custom_model.properties


# In[82]:


custom_model.submodels


# In[83]:


for submodel in custom_model.submodels:
    print(
        "The submodel with form type '{}' has recognized the following fields: {}".format(
            submodel.form_type, 
            ", ".join(
                {
                    field.label if field.label else name
                    for name, field in submodel.fields.items()
                }
            ),
        )
    )


# In[84]:


custom_model.model_id


# In[85]:


custom_model_info = form_training_client.get_custom_model(model_id=custom_model.model_id)
print("Model ID: {}".format(custom_model_info.model_id))
print("Status: {}".format(custom_model_info.status))
print("Training started on: {}".format(custom_model_info.training_started_on))
print("Training completed on: {}".format(custom_model_info.training_completed_on))


# In[86]:


new_test_url = "https://project1step02.blob.core.windows.net/boarding/boarding_pass-kamala-harris.pdf?sp=racwdyti&st=2022-06-05T02:59:47Z&se=2022-06-07T10:59:47Z&skoid=5f0cb329-aa4f-4571-8f0a-8eb5af801702&sktid=2b5b2a7c-d342-454f-b5d3-69ca89275ad2&skt=2022-06-05T02:59:47Z&ske=2022-06-07T10:59:47Z&sks=b&skv=2020-08-04&spr=https&sv=2020-08-04&sr=b&sig=RtnZArLMw5msDLUt1eLTVKEsQ0OPj5xvqznuWmhl%2B6Q%3D"


# In[108]:


show_image_in_cell(new_test_url)


# In[ ]:




