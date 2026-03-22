# 🎸 Spotify Genre Predictor (AI ทายแนวเพลง)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://musicthebestoftheworld-xpj4qdgc27pan4rfce8jdt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

Web Application สำหรับพยากรณ์แนวเพลง (Genre) จากลักษณะทางดนตรี (Audio Features) โดยใช้โมเดล Machine Learning อย่าง **Support Vector Machine (SVM)** ซึ่งเรียนรู้จากชุดข้อมูลเพลงบน Spotify

**https://musicthebestoftheworld-xpj4qdgc27pan4rfce8jdt.streamlit.app/**

---

## 🌟 ฟีเจอร์หลัก (Features)
ผู้ใช้งานสามารถปรับแต่งสไลเดอร์เพื่อจำลองสไตล์ดนตรีที่ต้องการ แล้ว AI จะทำการวิเคราะห์และทายผลออกมาเป็นแนวเพลงต่างๆ โดยอาศัยฟีเจอร์เสียง 9 มิติ:
- **Danceability:** ความเหมาะสมในการเต้นของเพลง
- **Energy:** พลังงาน ความหนักแน่น และความรวดเร็ว
- **Loudness:** ความดังของเพลง (หน่วยเป็นเดซิเบล dB)
- **Speechiness:** ปริมาณเสียงพูดหรือคำร้องในเพลง
- **Acousticness:** ปริมาณเสียงเครื่องดนตรีอะคูสติก
- **Instrumentalness:** สัดส่วนของเพลงที่ไม่มีเสียงร้อง
- **Liveness:** โอกาสที่เพลงนี้จะเป็นการแสดงสด
- **Valence:** ความร่าเริงหรืออารมณ์เชิงบวกของเพลง
- **Tempo:** ความเร็วของจังหวะเพลง (BPM)

---

## 🧠 ข้อมูลเกี่ยวกับโมเดล (Model Info)
โปรเจคนี้ได้ทำการเปรียบเทียบโมเดลหลายตัวและพบว่า **Support Vector Machine (SVC)** ให้ผลลัพธ์ที่มีประสิทธิภาพดีที่สุด
- **Algorithm:** Support Vector Classifier (RBF Kernel)
- **Data Preprocessing:** `StandardScaler` และ `LabelEncoder`

---
