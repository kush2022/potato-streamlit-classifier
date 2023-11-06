import tensorflow as tf 
import streamlit as st 
import numpy as np 
from PIL import Image
import datetime, time
import pandas as pd 

# load the model 
model = tf.keras.models.load_model("potato-disease.hv")


# define a function for making predictions 
def predict_leaf_disease(image):
    # preprocess the image
    img = image.resize((256, 256))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255 

    # make prediction 
    predictions = model.predict(img)

    # map the models out to class label
    class_labels = ['Potato Early blight', 'Potato Late blight', 'Potato Healthy']
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)


    return predicted_class, confidence


# Streamlit app layout 

st.set_page_config(
    page_title="Potato Disease",
    page_icon="🥔",
    layout="centered"
)


st.title("Potato Leaf Disease Classification")
st.subheader("🌱🍠 Unearth the power of AI in agriculture with our Potato Disease Classifier! 🍠🌱")
st.divider()
st.subheader("Upload a leaf image to classify the disease")


# file upload  

uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded image", use_column_width=False)

    # get prediction 
    col1, col2 = st.columns(2, gap="small")
    prediction_class, confidence = predict_leaf_disease(image)

    with st.spinner('Classifying for it...'):
        time.sleep(5)
        with col1:
            st.subheader("Predicted Disease:")
        
            st.write(f"{prediction_class}")
        with col2:
            st.subheader("Confidence:")
        
            st.write(f"{confidence:.2%}")





hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            div.embeddedAppMetaInfoBar_container__DxxL1 {visibility: hidden;}
            background-image: url("/home/felix/Desktop/ML-Projects/Potato-Disease/11 Things Humans Do That Dogs Hate.jpg");
background-size: cover; 
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)