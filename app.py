import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load model and labels
model = tf.keras.models.load_model('keras_model.h5')

with open('labels.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f.readlines()]

# Load allergens CSV
allergens_df = pd.read_csv('menu_with_allergens.csv')
allergens_df['Menu'] = allergens_df['Menu'].str.lower().str.strip()

st.title("📷 Allergic-Eye")

# Camera input
camera_image = st.camera_input("사진을 찍어 주세요!")

if camera_image is not None:
    # 이미지 처리
    image = Image.open(io.BytesIO(camera_image.getvalue())).convert('RGB').resize((224, 224))
    st.image(image, caption="촬영한 이미지", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 모델 예측
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_food = labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.subheader(f"🍔 예측된 음식: **{predicted_food}**")
    st.write(f"📈 신뢰도: **{confidence:.2f}%**")

    # 알러지 정보 찾기
    # 알러지 정보 무조건 출력
    matching_rows = allergens_df[allergens_df['Menu'] == predicted_food.lower()]

    allergens = matching_rows['Allergens'].values[0]
    st.warning(f"⚠️ 알러지 성분: **{allergens}**")



