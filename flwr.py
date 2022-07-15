# %%
import streamlit as st
import os
import cv2
from PIL import Image,ImageOps
import numpy as np
from tensorflow import keras
from keras.utils.data_utils import get_file

# %%
# model=keras.models.load_model('https://drive.google.com/file/d/1qXbkuRnj9ZGqWxeHVccJSStp3LxZGjA9/view?usp=sharing')   

st.write("""
            # FLOWER CLASSIFICATION
         """)
#getting the local file path and the url
model=keras.models.load_model('Flower_prediction_new.h5')
pic_upload=st.file_uploader('Upload only flower Image',type=['jpg','png'])

#resizing the image
if pic_upload is not None:
    #convert the file into opencv image
    file_bytes=np.asarray(bytearray(pic_upload.read()),dtype=np.uint8)
    opencv_img=cv2.imdecode(file_bytes,1)  
    opencv_img=cv2.cvtColor(opencv_img,cv2.COLOR_BGR2RGB)
    opencv_img=cv2.resize(opencv_img,(200,200))
    #displaying the image
    st.image(pic_upload,channels='RGB',width=400)
    
    final_format=np.array([opencv_img]).astype('float64')/255.0
    #using our trained model to predict the input
    pred=model.predict(final_format)
    index=np.argmax(pred[0])
    labels_names=['daisy','dandelion','roses','sunflower','tulips']

    porob=round(np.max(pred[0]*100),2)

    label=labels_names[index]

    st.title('The flower is {} {}'.format(label,porob))

# %%


# %%
