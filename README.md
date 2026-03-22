# 🎧 Music Genre AI Predictor & Recommender

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://musicthebestoftheworld-xpj4qdgc27pan4rfce8jdt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

Web Application สุดล้ำที่ผสมผสาน **AI พยากรณ์แนวเพลง** เข้ากับ **ระบบแนะนำเพลงยอดฮิต** มาพร้อมกับดีไซน์ UI ที่สวยงามทันสมัย (Netflix-Inspired) 

👉 **[คลิกที่นี่เพื่อลองเล่น Web App ของเรา!](https://musicthebestoftheworld-xpj4qdgc27pan4rfce8jdt.streamlit.app/)**

---

## ✨ ฟีเจอร์เด่น (Key Features)

1. 🤖 **AI Genre Prediction:** ทายแนวเพลงที่คุณชอบได้อย่างแม่นยำด้วยโมเดล Machine Learning (SVM) จากการปรับสไลเดอร์ค่าต่างๆ เช่น ความเร็ว (Tempo), ความสนุก (Danceability), เสียงร้อง (Speechiness) เป็นต้น
2. 🎶 **Smart Song Recommendation:** เมื่อ AI ทายแนวเพลงของคุณได้แล้ว ระบบจะดึง **"5 เพลงฮิตยอดนิยม"** ในแนวเพลงนั้นๆ พร้อมชื่อศิลปินตัวจริง มาแนะนำให้คุณไปฟังต่อทันที!
3. 🎨 **Sleek User Interface:** หน้าตาเว็บถูกออกแบบใหม่ให้ใช้งานง่าย แบ่งสัดส่วนชัดเจน และมีสีสันสไตล์สตรีมมิ่งแพลตฟอร์มระดับโลก

---

## 🧠 เบื้องหลังการทำงาน (How it works)

- **Machine Learning Model:** เราใช้โมเดล **Support Vector Machine (SVC)** คู่กับ RBF Kernel ซึ่งพิสูจน์แล้วว่าทำงานได้ดีที่สุดกับชุดข้อมูลนี้ โดยโมเดลถูกบีบอัดให้มีขนาดเล็กเพื่อการประมวลผลบนเว็บที่รวดเร็ว
- **Dataset:** โมเดลเรียนรู้และดึงข้อมูลเพลงแนะนำมาจากชุดข้อมูล Spotify Tracks จำนวนมหาศาล
- **Data Preprocessing:** ข้อมูลถูกจัดการด้วย `StandardScaler` (ปรับสเกล) และ `LabelEncoder` (แปลงคลาสแนวเพลง)

---

## 📂 โครงสร้างไฟล์ในโปรเจค (Repository Structure)

```text
├── app.py                   # โค้ดหลักสำหรับหน้าเว็บ (UI + Model + Recommendation)
├── svm_model.pkl            # ไฟล์สมอง AI (SVM Model) ที่บีบอัดแล้ว
├── scaler.pkl               # ไฟล์ตัวปรับสเกลข้อมูล
├── label_encoder.pkl        # ไฟล์ถอดรหัสชื่อแนวเพลง
├── spotify_tracks.csv       # ฐานข้อมูลเพลงสำหรับใช้แนะนำศิลปินและเพลงฮิต (สำคัญ!)
├── requirements.txt         # รายชื่อ Library (streamlit, pandas, scikit-learn, joblib)
└── Phomtummaipen.ipynb      # Source Code ขั้นตอนการเทรนโมเดล
