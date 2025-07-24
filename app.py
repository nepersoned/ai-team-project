import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import io

model = tf.keras.models.load_model('keras_model.h5')
with open('labels.txt', 'r', encoding='utf-8') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

allergens_df = pd.read_csv('menu_with_allergens.csv', encoding='utf-8')

def clean_text(text):
    return str(text).replace(" ", "").strip()

allergens_df['Cleaned_Menu'] = allergens_df['Menu'].apply(clean_text)

st.title("📷 Allergic-Eye")

camera_image = st.camera_input("사진을 찍어 주세요!")

if camera_image is not None:
    image = Image.open(io.BytesIO(camera_image.getvalue())).convert('RGB').resize((224, 224))
    st.image(image, caption="촬영한 이미지", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_food = labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    
    if confidence > 85 and predicted_food != '없음':
        st.subheader(f"🥄 예측된 음식: **{predicted_food}**")
        st.write(f"📈 신뢰도: **{confidence:.2f}%**")
    else:
        st.error(f"❌ 음식을 특정할수 없습니다.")
        
    cleaned_predicted_food = clean_text(predicted_food)
    matching_rows = allergens_df[allergens_df['Cleaned_Menu'] == cleaned_predicted_food]

    if not matching_rows.empty and confidence > 85 and predicted_food != '없음':
        allergens = matching_rows['Allergens'].values[0]
        calories = matching_rows['Calories'].values[0]
        cal_number = int(''.join(filter(str.isdigit, calories)))
        if cal_number <= 500:
            cal_level = "저칼로리"
            cal_icon = "🟢"
        elif cal_number <= 800:
            cal_level = "중간칼로리"
            cal_icon = "🟡"
        else:
            cal_level = "고칼로리"
            cal_icon = "🔴"

        st.warning(f"⚠ 알러지 성분: **{allergens}**") 
        st.info(f"🔥 칼로리: {cal_icon} **{calories} ({cal_level})**")

