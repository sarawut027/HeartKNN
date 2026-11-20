from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="โปรเจคข้อมูลโรคหัวใจ",
    page_icon="❤️",
    layout="wide"
)

# ----------------- Modern Healthcare CSS Styling (เปลี่ยนเฉพาะสไตล์) -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* Main app background with modern clean medical gradient */
.stApp {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 30%, #EFF6FF 70%, #FFF1F2 100%);
    background-attachment: fixed;
    font-family: 'Prompt', sans-serif;
    color: #1E293B;
}

/* Main Header */
h1 {
    font-family: 'Prompt', sans-serif !important;
    color: #BE123C !important;
    text-align: center;
    font-weight: 700 !important;
    padding-top: 10px;
    padding-bottom: 20px;
    letter-spacing: 0.5px;
    text-shadow: 1px 2px 4px rgba(190, 18, 60, 0.15);
}

/* Sub-headers inside cards */
[data-testid="column"] h2 {
    font-family: 'Prompt', sans-serif !important;
    color: #9F1239 !important;
    text-align: center;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    margin-bottom: 12px;
}

/* Modern Glassmorphic Card for columns */
[data-testid="column"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 20px 16px;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(225, 29, 72, 0.15);
    transition: all 0.35s ease;
}

[data-testid="column"]:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 32px rgba(190, 18, 60, 0.2);
    border-color: #FDA4AF;
}

/* Modern styling for images */
[data-testid="stImage"] img {
    border-radius: 14px;
    object-fit: cover;
    width: 100%;
    max-height: 250px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
    transition: transform 0.3s ease;
}

[data-testid="stImage"] img:hover {
    transform: scale(1.02);
}

/* Dataframe container */
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    border: 1px solid rgba(226, 232, 240, 0.8);
}

/* Modern Cardiology Buttons */
.stButton > button {
    background: linear-gradient(135deg, #E11D48 0%, #BE123C 100%) !important;
    color: white !important;
    font-family: 'Prompt', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 30px !important;
    border: none !important;
    padding: 0.6rem 2.2rem !important;
    box-shadow: 0 8px 20px rgba(225, 29, 72, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 26px rgba(225, 29, 72, 0.45) !important;
    background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%) !important;
    color: white !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Inputs & Sliders Cards */
.stSlider, .stNumberInput {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(6px);
    padding: 14px 20px;
    border-radius: 16px;
    border: 1px solid rgba(226, 232, 240, 0.8);
    margin-bottom: 14px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
}

div[data-baseweb="slider"] {
    margin-top: 10px;
}

div[data-baseweb="input"] {
    border-radius: 12px !important;
    border-color: #CBD5E1 !important;
}

.stAlert {
    border-radius: 14px !important;
}

.stMarkdown p {
    font-family: 'Prompt', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Title Header -----------------
st.header("💓 โปรเจคการจำแนกโรคหัวใจ 💓")
st.image("./img/Sarawut.jpg")

# ----------------- 3 Species Showcase -----------------
col1, col2 = st.columns(2)

with col1:
    st.header("หัวใจปกติ (No Heart Disease)")
    st.image("./img/good.png")

with col2:
    st.header("ผู้ป่วยโรคหัวใจ (Heart Disease)")
    st.image("./img/bad.png")


# ----------------- Data Section -----------------
html_7 = """
<div style="background: linear-gradient(135deg, #E11D48 0%, #BE123C 50%, #9F1239 100%); padding: 16px 20px; border-radius: 18px; box-shadow: 0 8px 22px rgba(190, 18, 60, 0.25); margin: 24px 0 16px 0; text-align: center;">
    <h4 style="color: white; margin: 0; font-family: 'Prompt', sans-serif; font-weight: 600; letter-spacing: 0.5px;">🌸 สถิติข้อมูลดอกไม้ 🌸</h4>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))

dt1 = dt['Age'].mean()
dt2 = dt['Cholesterol'].mean()
dt3 = dt['MaxHR'].mean()
dt4 = dt['RestingBP'].mean()

dx = [dt1, dt2, dt3, dt4]
dx2 = pd.DataFrame(dx, index=["Age", "Cholesterol", "MaxHR", "RestingBP"])

if st.button("แสดงการจินตทัศน์ข้อมูล"):
   #st.write(dt.head(10))
   st.bar_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

# ----------------- Prediction Section -----------------
html_8 = """
<div style="background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #BE123C 100%); padding: 16px 20px; border-radius: 18px; box-shadow: 0 8px 22px rgba(2, 132, 199, 0.25); margin: 28px 0 16px 0; text-align: center;">
    <h4 style="color: white; margin: 0; font-family: 'Prompt', sans-serif; font-weight: 600; letter-spacing: 0.5px;">🌺 ทำนายข้อมูล 🌺</h4>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")


age = st.slider("กรุณาเลือกอายุ (Age)", min_value=18, max_value=100, value=50)
resting_bp = st.slider("กรุณาเลือกความดันโลหิตขณะพัก (RestingBP)", min_value=80, max_value=200, value=120)
chol = st.number_input("กรุณากรอกระดับคอเลสเตอรอล (Cholesterol)", min_value=100, max_value=600, value=200)
max_hr = st.number_input("กรุณากรอกอัตราการเต้นหัวใจสูงสุด (MaxHR)", min_value=60, max_value=220, value=150)
if st.button("ทำนายผล"):
   #st.write("ทำนาย")
   #dt = pd.read_csv("./data/iris.csv") 

   # เลือก 4 ฟีเจอร์ที่ตรงกับช่องรับข้อมูลด้านบน
   features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
   X = dt[features]
   y = dt['HeartDisease']          # ✅ เป้าหมายคือ 0 (ปกติ) หรือ 1 (เป็นโรค)

   Knn_model = KNeighborsClassifier(n_neighbors=3)
   Knn_model.fit(X, y)

   x_input = np.array([[age, resting_bp, chol, max_hr]]) #ข้อมูลใหม่สำหรับทำนาย

   st.write(Knn_model.predict(x_input))   
   out=Knn_model.predict(x_input)

   if out[0] == 0:
       st.success("ผลการวิเคราะห์: สุขภาพหัวใจปกติ (Normal)")
       st.image("./img/good.png")  # หรือเปลี่ยนเป็นรูปหัวใจปกติ
   else:
       st.error("ผลการวิเคราะห์: มีความเสี่ยงเป็นโรคหัวใจ (Heart Disease Detected)")
       st.image("./img/bad.png")  # หรือเปลี่ยนเป็นรูปหัวใจมีปัญหา
else:
   st.write("ไม่ทำนาย")
