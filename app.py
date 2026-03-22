import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Aesthetic Upgrade สไตล์ )
# ==========================================
st.set_page_config(
    page_title=" Music Genre AI Predictor | The Best of The World", 
    page_icon="🎧",  
    layout="wide" 
)

# ==========================================
# 2. ฟังก์ชันโหลด Assets และ ข้อมูลเพลง
# ==========================================
@st.cache_resource
def load_ml_assets():
    """โหลดโมเดล AI ที่ทำไว้แล้ว"""
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
        return model, scaler, le
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบการอัปโหลดไฟล์ .pkl ทั้ง 3 ไฟล์")
        return None, None, None

@st.cache_data
def load_raw_data():
    """โหลดข้อมูล CSV ตัวจริง เพื่อใช้ค้นหาเพลงมาแนะนำ"""
    try:
        df = pd.read_csv('spotify_tracks.csv')
        # ตรวจสอบหาคอลัมน์ genre
        possible_genre_cols = ['genre', 'track_genre', 'playlist_genre']
        found_col = None
        for col in possible_genre_cols:
            if col in df.columns:
                found_col = col
                break
        
        if found_col and found_col != 'genre':
            df = df.rename(columns={found_col: 'genre'})
            
        return df.dropna()
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ spotify_tracks.csv! ระบบแนะนำเพลงจะไม่ทำงาน")
        return None

# เรียกใช้ฟังก์ชันโหลด
model, scaler, le = load_ml_assets()
raw_df = load_raw_data()

# ==========================================
# 3. ส่วนหัวของแอป (UI)
# ==========================================
st.markdown("""
<style>
    .big-header {
        background-color: #1DB954; /* สีเขียว Spotify */
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
    }
</style>
<div class="big-header">
    <h1>🎧 Musi Music Predictor</h1>
    <p>ทำนายแนวเพลงและแนะนำเพลง</p>
</div>
""", unsafe_allow_html=True)

# ตรวจสอบความพร้อมของโมเดล
if model is not None and le is not None:
    # แบ่งหน้าจอเป็น 2 คอลัมน์
    col_input, col_result = st.columns([1, 1.5])

    # ==========================================
    # 4. คอลัมน์ซ้าย: ส่วนรับข้อมูล (Inputs)
    # ==========================================
    with col_input:
        with st.form("music_features_form"):
            st.subheader("📊 ปรับแต่งลักษณะทางดนตรีของคุณ")
            
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                danceability = st.slider("ความสนุกในการเต้น (Danceability)", 0.0, 1.0, 0.5)
                energy = st.slider("พลังงานของเพลง (Energy)", 0.0, 1.0, 0.5)
                loudness = st.number_input("ความดัง (Loudness dB)", -60.0, 0.0, -10.0)
                speechiness = st.slider("ปริมาณคำพูด (Speechiness)", 0.0, 1.0, 0.1)
                
            with form_col2:
                acousticness = st.slider("เสียงอะคูสติก (Acousticness)", 0.0, 1.0, 0.2)
                instrumentalness = st.slider("เสียงเครื่องดนตรีล้วน (Instrumentalness)", 0.0, 1.0, 0.0)
                valence = st.slider("ความร่าเริงของอารมณ์เพลง (Valence)", 0.0, 1.0, 0.5)
                tempo = st.number_input("จังหวะ BPM (Tempo)", 50.0, 250.0, 120.0)
                liveness = 0.1 # ค่าคงที่เพื่อประหยัดพื้นที่หน้าจอ

            submit = st.form_submit_button("🌟 วิเคราะห์และแนะนำเพลง")

    # ==========================================
    # 5. คอลัมน์ขวา: ผลลัพธ์และการแนะนำเพลง
    # ==========================================
    with col_result:
        if not submit:
            # ก่อนกดปุ่ม ให้แสดงรูปภาพต้อนรับ
            st.info("👈 ปรับจังหวะดนตรีด้านซ้าย แล้วกดปุ่มเพื่อดูผลลัพธ์!")
            st.image("My_pic.jpg", caption="แอปแนะนำเพลงของฉัน 🎧", width=300)
            
        else:
            # เมื่อกดปุ่ม วิเคราะห์และทายผล
            input_data = np.array([[danceability, energy, loudness, speechiness, 
                                    acousticness, instrumentalness, liveness, valence, tempo]])
            
            # กรองข้อมูลผ่าน Scaler และทำนายผล
            input_scaled = scaler.transform(input_data)
            predicted_num = model.predict(input_scaled)[0]
            predicted_genre = le.inverse_transform([predicted_num])[0]
            
            probabilities = model.predict_proba(input_scaled)[0]
            max_prob = max(probabilities) * 100

            # --- โชว์ผลการทายแนวเพลง ---
            st.markdown(f"""
            <div style='background-color: #282828; color: white; padding: 20px; border-radius: 15px; border-left: 5px solid #1DB954;'>
                <h3>🎉 แนวเพลงของคุณคือ: <span style='color: #1DB954; font-size: 1.2em;'>{predicted_genre.upper()}</span></h3>
                <p>AI มั่นใจ: <b>{max_prob:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.divider()

            # --- ฟังก์ชันแนะนำเพลง ---
            st.subheader(f"🎶 เพลง {predicted_genre.upper()} ยอดฮิตที่ AI แนะนำ")
            
            if raw_df is not None:
                # ค้นหาเพลงที่ตรงกับแนวเพลงที่ทายได้
                recommended_songs = raw_df[raw_df['genre'] == predicted_genre]
                
                if not recommended_songs.empty:
                    # เรียงตาม popularity ถ้ามี
                    if 'popularity' in recommended_songs.columns:
                        top_songs = recommended_songs.sort_values(by='popularity', ascending=False).head(5)
                    else:
                        top_songs = recommended_songs.sample(n=min(5, len(recommended_songs)))

                    # --- โชว์เพลงแนะนำแบบเป็นการ์ด ---
                    st.markdown("""
                    <style>
                        .song-card {
                            background-color: #f0f2f6;
                            padding: 10px;
                            border-radius: 8px;
                            margin-bottom: 8px;
                            color: #282828;
                        }
                    </style>
                    """, unsafe_allow_html=True)

                    for index, row in top_songs.iterrows():
                        track_name = row['track_name'] if 'track_name' in raw_df.columns else "Unknown Track"
                        # แก้ไขชื่อคอลัมน์ศิลปินเป็น artist_name ให้ตรงกับข้อมูล
                        artists = row['artist_name'] if 'artist_name' in raw_df.columns else "Unknown Artist"
                        
                        st.markdown(f"""
                        <div class="song-card">
                            <b>🎵 {track_name}</b><br>
                            <span style='font-size: 0.9em;'>โดย {artists}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"❌ ขออภัย ไม่พบเพลงแนว '{predicted_genre}' ในระบบ")

st.divider()
