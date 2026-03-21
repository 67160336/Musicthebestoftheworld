import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Spotify Genre Predictor", page_icon="🎸", layout="centered")

# 2. ฟังก์ชันโหลด Model, Scaler และ LabelEncoder
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบการอัปโหลดไฟล์ .pkl ทั้ง 3 ไฟล์")
        return None, None, None

model, scaler, le = load_assets()

# 3. ส่วนหัวของแอป
st.title("🎸 AI ทายแนวเพลง (SVM Model)")
st.write("ปรับสไลเดอร์เพื่อสร้างสไตล์ดนตรี แล้วดูว่า AI (Support Vector Machine) จะทายว่าเป็นเพลงแนวไหน!")

if model is not None:
    # 4. ส่วนรับข้อมูล
    with st.form("music_features_form"):
        st.subheader("📊 ปรับแต่งลักษณะทางดนตรี")
        col1, col2 = st.columns(2)
        
        with col1:
            danceability = st.slider("Danceability (เต้นตามได้)", 0.0, 1.0, 0.5)
            energy = st.slider("Energy (ความหนักแน่น)", 0.0, 1.0, 0.5)
            loudness = st.number_input("Loudness (ความดัง dB)", -60.0, 0.0, -10.0)
            speechiness = st.slider("Speechiness (ปริมาณคำพูด)", 0.0, 1.0, 0.1)
            
        with col2:
            acousticness = st.slider("Acousticness (เสียงอะคูสติก)", 0.0, 1.0, 0.2)
            instrumentalness = st.slider("Instrumentalness (ความไม่มีเสียงร้อง)", 0.0, 1.0, 0.0)
            valence = st.slider("Valence (อารมณ์เพลง)", 0.0, 1.0, 0.5)
            tempo = st.number_input("Tempo (จังหวะ BPM)", 50.0, 250.0, 120.0)
            liveness = 0.1 

        submit = st.form_submit_button("🎧 วิเคราะห์แนวเพลง")

    # 5. ส่วนแสดงผล
    if submit:
        input_data = np.array([[danceability, energy, loudness, speechiness, 
                                acousticness, instrumentalness, liveness, valence, tempo]])
        
        # กรองข้อมูลผ่าน Scaler
        input_scaled = scaler.transform(input_data)
        
        # ทำนายผล (ได้ออกมาเป็นตัวเลข)
        predicted_num = model.predict(input_scaled)[0]
        
        # แปลงตัวเลขกลับเป็นชื่อแนวเพลง
        predicted_genre = le.inverse_transform([predicted_num])[0]
        
        # ดึงความน่าจะเป็น (เปอร์เซ็นต์ความมั่นใจ)
        probabilities = model.predict_proba(input_scaled)[0]
        max_prob = max(probabilities) * 100

        st.divider()

        # แสดงผลแนวเพลง
        st.success(f"### 🎉 แนวเพลงของคุณคือ: **{predicted_genre.upper()}**")
        st.write(f"ความมั่นใจของโมเดล SVM: **{max_prob:.2f}%**")
        
        # แสดงแนวเพลงรองลงมา
        st.caption("แนวเพลงอื่นๆ ที่มีความเป็นไปได้:")
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        for i in top3_indices:
            genre_name = le.inverse_transform([i])[0]
            if genre_name != predicted_genre:
                st.write(f"- {genre_name}: {probabilities[i]*100:.2f}%")

st.divider()
