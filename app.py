import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io

model = tf.keras.models.load_model('keras_model.h5')

with open('labels.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip().split(maxsplit=1)[1].strip() for line in f.readlines()]

allergens_df = pd.read_csv('menu_with_allergens (1).csv')
allergens_df['Menu'] = allergens_df['Menu'].str.lower().str.strip()  # 소문자, 공백 정리

st.title("📷 Allergic-Eye")


camera_image = st.camera_input("사진을 찍어 주세요!")

if camera_image is not None:
    image = Image.open(io.BytesIO(camera_image.getvalue())).convert('RGB').resize((224, 224))
    st.image(image, caption="촬영한 이미지", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_food = labels[predicted_index].lower().strip()  # 소문자, 공백 제거
    confidence = predictions[0][predicted_index] * 100

    st.subheader(f"🍔 예측된 음식: **{predicted_food}**")
    # st.write(f"📈 신뢰도: **{confidence:.2f}%**")  # 필요시 주석 해제

    allergens = allergens_df[allergens_df['Menu'] == predicted_food]['Allergens'].values[0]
    st.warning(f"⚠️ 알러지 성분: **{allergens}**")
