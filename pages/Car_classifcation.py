import streamlit as st
import pytesseract
from matplotlib import pyplot as plt
import numpy as np
import keras
from PIL import Image, ImageOps

st.write("## 🚗 🛻차종 분류 홈페이지입니다!🚗 🛻")
st.write("### K3, K5, 싼타페, 투싼, 그렌저, 소나타, 니로만 가능합니다.")


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction)  # return position of the highest probability


image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

if image_file is not None:
    # st.image(plt.imread(image_file))
    image = Image.open(image_file)
    st.image(image, caption='Uploaded Car Image.')
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'model/keras_model.h5')
    if label == 0:
        st.subheader("KIA_SUV 니로")
    elif label == 1:
        st.subheader("현대_SUV 산타페")
    elif label == 2:
        st.subheader("현대_SUV 투싼")
    elif label == 3:
        st.subheader("KIA_세단 K3")
    elif label == 4:
        st.subheader("KIA_세단 K5")
    elif label == 5:
        st.subheader("현대_세단 그렌저")
    else:
        st.subheader("현대_세단 쏘나타")
