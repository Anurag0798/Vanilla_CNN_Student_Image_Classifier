import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = 'Vanilla_CNN_Student_Image_Classifier.h5'

model = load_model(MODEL_PATH)

class_names = ['anu', 'bharti', 'deepak', 'manidhar', 'sudh']

st.set_page_config(page_title = "Student Image Classification", layout = 'centered')

st.sidebar.title("Upload your image")
st.markdown("This applicatoin will classify your image. It's built on a Vanilla CNN architecture")

upload_file = st.sidebar.file_uploader("Upload your image", type = ["jpg", 'jpeg', 'png'])

if upload_file is not None :
    img = Image.open(upload_file).convert('RGB')
    st.image(img, caption="Your uploaded image")
    
    image_resized = img.resize((128,128))
    img_array = image.img_to_array(image_resized) / 255.0
    image_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(image_batch)
    predicted_class = class_names[np.argmax(prediction)]
    
    st.success(f"This image is predicted to be : {predicted_class}")
    
    st.subheader("Below is the confidence score for all the classes")
    
    print(prediction)
    
    for index, score in enumerate(prediction[0]):
        st.write(f"{class_names[index]} : {score}")