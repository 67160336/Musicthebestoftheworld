import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Aesthetic Upgrade เหมือน Netflix)
# ==========================================
st.set_page_config(
    page_title="Music Genre AI | The Best of The World", 
    page_icon="🎧",  # ใช้ไอคอนหูฟังแทน🎸
    layout="wide" # ปรับ layout เป็น wide เพื่อให้มีพื้นที่โชว์เพลงแนะนำ
)

# ==========================================
# 2. ฟังก์ชันโหลด Assets (เพิ่มการโหลดข้อมูล CSV)
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
        st.error("❌ ไม่พบไฟล์โมเดล SVM! กรุณาตรวจสอบการอัปโหลดไฟล์ .pkl")
        return None, None, None

@st.cache_data
def load_raw_data():
    """โหลดข้อมูล CSV ตัวจริง เพื่อใช้ค้นหาเพลงมาแนะนำ (จุดที่เพิ่มใหม่!)"""
    try:
        df = pd.read_csv('spotify_tracks.csv')
        # ตรวจสอบว่ามีคอลัมน์ genre หรือไม่ ถ้าไม่มีให้พยายามหาคอลัมน์อื่นที่ใกล้เคียง
        possible_genre_cols = ['genre', 'track_genre', 'playlist_genre']
        found_col = None
        for col in possible_genre_cols:
            if col in df.columns:
                found_col = col
                break
        
        if found_col and found_col != 'genre':
            df = df.rename(columns={found_col: 'genre'})
            
        return df.dropna() # ลบค่าว่าง
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ spotify_tracks.csv! การแนะนำเพลงจะไม่ทำงาน")
        return None

# เรียกใช้ฟังก์ชันโหลด
model, scaler, le = load_ml_assets()
raw_df = load_raw_data()

# ==========================================
# 3. ส่วนหัวของแอป (UI สไตล์ Netflix)
# ==========================================
# สร้าง container สีเขียวเข้มเพื่อเป็น Header สวยๆ
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
    <h1>🎧 Music Genre AI Predictor</h1>
    <p>Predict genre and discover top hits in that style!</p>
</div>
""", unsafe_allow_html=True)

# ตรวจสอบความพร้อม
if model is not None and le is not None:
    # แบ่งหน้าจอเป็น 2 คอลัมน์ (เหมือน Netflix App)
    col_input, col_result = st.columns([1, 1.5]) # คอลัมน์ขวา (ผลลัพธ์) กว้างกว่า

    # ==========================================
    # 4. คอลัมน์ซ้าย: ส่วนรับข้อมูล (Inputs)
    # ==========================================
    with col_input:
        with st.form("music_features_form"):
            st.subheader("📊 ปรับแต่งลักษณะทางดนตรีของคุณ")
            
            # ใช้ st.columns ข้างใน form เพื่อจัดกลุ่มสไลเดอร์ให้สวยงาม
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
    # 5. คอลัมน์ขวา: ผลลัพธ์และการแนะนำเพลง (Results & Recs)
    # ==========================================
    with col_result:
        if not submit:
            # ก่อนกดปุ่ม ให้แสดงรูปภาพหรือข้อความต้อนรับ (Placeholder เหมือน Netflix)
            st.info("👈 ปรับจังหวะดนตรีด้านซ้าย แล้วกดปุ่มเพื่อดูผลลัพธ์!")
            st.image("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=600&auto=format&fit=crop", caption="Photo by Unsplash")
            
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

            # --- โชว์ผลการทายแนวเพลง (Prediction Card) ---
            st.markdown(f"""
            <div style='background-color: #282828; color: white; padding: 20px; border-radius: 15px; border-left: 5px solid #1DB954;'>
                <h3>🎉 แนวเพลงของคุณคือ: <span style='color: #1DB954; font-size: 1.2em;'>{predicted_genre.upper()}</span></h3>
                <p>AI มั่นใจ: <b>{max_prob:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.divider()

            # ==========================================
            # 6. ฟังก์ชันแนะนำเพลง (Recommendations - จุดที่เพิ่มใหม่!)
            # ==========================================
            st.subheader(f"🎶 เพลง {predicted_genre.upper()} ยอดฮิตที่ AI แนะนำ")
            
            if raw_df is not None:
                # ค้นหาเพลงที่ตรงกับแนวเพลงที่ทายได้
                recommended_songs = raw_df[raw_df['genre'] == predicted_genre]
                
                if not recommended_songs.empty:
                    # สมมติว่ามีคอลัมน์ popularity (ความนิยม) เราจะเลือกเพลงที่ฮิตที่สุด 5 เพลง
                    # หากไม่มีคอลัมน์ popularity โค้ดจะสุ่มเพลงมาแทน
                    if 'popularity' in recommended_songs.columns:
                        top_songs = recommended_songs.sort_values(by='popularity', ascending=False).head(5)
                    else:
                        st.caption("*(ไม่พบข้อมูลความนิยมของเพลง จึงทำการสุ่มเพลงมาแทน)*")
                        top_songs = recommended_songs.sample(n=min(5, len(recommended_songs)))

                    # --- โชว์เพลงแนะนำแบบเป็นการ์ด (สวยเหมือน Netflix) ---
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
                        artists = row['artists'] if 'artists' in raw_df.columns else "Unknown Artist"
                        
                        st.markdown(f"""
                        <div class="song-card">
                            <b>🎵 {track_name}</b><br>
                            <span style='font-size: 0.9em;'>โดย {artists}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(f"❌ ขออภัย ไม่พบเพลงแนว '{predicted_genre}' ในชุดข้อมูลของเราเพื่อแนะนำ")
            else:
                st.warning("⚠️ การแนะนำเพลงไม่ทำงาน เนื่องจากโหลดไฟล์ spotify_tracks.csv ไม่ได้")

st.divider()
