import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# 모델과 라벨 불러오기
model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# 알러지 CSV 불러오기
allergens_df = pd.read_csv('menu_with_allergens.csv', encoding='utf-8')

# 한글 매칭을 위한 정제 함수
def clean_text(text):
    return str(text).replace(" ", "").strip()

# CSV에 정제된 컬럼 추가
allergens_df['Cleaned_Menu'] = allergens_df['Menu'].apply(clean_text)

st.title("📷 Allergic-Eye")

# 카메라 입력 받기
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

    # 예측값 정제 후 매칭
    cleaned_predicted_food = clean_text(predicted_food)
    matching_rows = allergens_df[allergens_df['Cleaned_Menu'] == cleaned_predicted_food]

    if confidence < 90:
        print('음식을 특정할수 없습니다.')

    elif  not matching_rows.empty:
        allergens = matching_rows['Allergens'].values[0]
        st.warning(f"⚠ 알러지 성분: **{allergens}**") 
