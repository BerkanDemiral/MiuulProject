import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import warnings
import plotly.graph_objects as go
import base64
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="BurnedCalories", page_icon="🧘‍♂️")


st.title(":orange[Calories]:blue[ Burned During]:red[ Exercise]")

@st.cache_data
def load_data():
    return pd.read_csv("dataset/lastDataset.csv")

data = load_data()

X = data.drop(columns=["Calories_Burned"])
y = data["Calories_Burned"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

le_gender = LabelEncoder()
le_workout_type = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])
X["Workout_Type"] = le_workout_type.fit_transform(X["Workout_Type"])

tab1, tab2, tab3 = st.tabs(["👴 Vücut Bilgileri", "🏄 Egzersiz Verileri", "🫀 Nabız Verileri"])

with tab1:
    st.header("Vücut Bilgileriniz")
    age = st.slider("Yaşınızı Seçin", min_value=18, max_value=80, value=25, step=1)
    height = st.slider("Boyunuzu Seçin (cm)", min_value=120, max_value=230, value=170, step=1)
    weight = st.slider("Kilonuzu Seçin (kg)", min_value=40, max_value=150, value=70, step=1)

    st.markdown("Cinsiyetinizi Seçin")
    col1, col2 = st.columns(2)

    if "gender" not in st.session_state:
        st.session_state["gender"] = None

    with col1:
        if st.button("Kadın", key="woman_button"):
            st.session_state["gender"] = 0
        st.image("images/woman.png", width=200, caption="Kadın")

    with col2:
        if st.button("Erkek", key="man_button"):
            st.session_state["gender"] = 1
        st.image("images/man.png", width=200, caption="Erkek")

    gender_value = st.session_state.get("gender", None)

with tab2:
    st.header("Egzersiz Verileriniz")
    exercise_duration = st.slider("Egzersiz Süresi (dakika)", min_value=0, max_value=300, value=60, step=1)

    st.subheader("Egzersiz Türünüzü Seçin")
    col1, col2, col3, col4 = st.columns(4)

    if "exercise_type" not in st.session_state:
        st.session_state["exercise_type"] = None

    with col1:
        if st.button("Cardio", key="cardio_button"):
            st.session_state["exercise_type"] = 0
        st.image("images/cardio.PNG", width=150, caption="Cardio")

    with col2:
        if st.button("HIIT", key="hiit_button"):
            st.session_state["exercise_type"] = 1
        st.image("images/hiit.PNG", width=150, caption="HIIT")

    with col3:
        if st.button("Strength", key="strength_button"):
            st.session_state["exercise_type"] = 2
        st.image("images/strength.PNG", width=150, caption="Strength")

    with col4:
        if st.button("Yoga", key="yoga_button"):
            st.session_state["exercise_type"] = 3
        st.image("images/yoga.PNG", width=150, caption="Yoga")

    exercise_type = st.session_state.get("exercise_type", None)

    if exercise_type is not None:
        workout_type_value = exercise_type
    else:
        workout_type_value = None

    water_intake = st.slider("Su İçtiyseniz Kaç Litre?", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

with tab3:
    st.header("Nabız Verileriniz")
    max_bpm = st.number_input("Maksimum Nabız", min_value=0, step=1, value=120)
    avg_bpm = st.number_input("Ortalama Nabız", min_value=0, step=1, value=80)
    resting_bpm = st.number_input("Dinlenme Nabzı", min_value=0, step=1, value=60)

    
    if st.button("Tahminleme Yap"):
        if gender_value is not None and workout_type_value is not None:
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            fat_percentage = 1.20 * bmi + 0.23 * age - (16.2 if gender_value == 1 else 5.4)

            
            input_data = np.array([[age, gender_value, weight, height_m, max_bpm, avg_bpm, resting_bpm,
                                    exercise_duration / 60, workout_type_value, fat_percentage, water_intake, bmi]])
            input_data_scaled = scaler.transform(input_data)

            
            predicted_calories = model.predict(input_data_scaled)[0]

            
            st.subheader("Tahmin Sonucu")
            st.write(f"Tahmini Yakılan Kalori: {predicted_calories:.2f} kalori")
            st.write(f"BMI: {bmi:.2f}, Yağ Oranı: {fat_percentage:.2f}%")

            
            if bmi < 18.5:
                st.success(
                    f"Egzersiz verileriniz incelendiğinde, **BMI** değeriniz **{bmi:.2f}**, yağ oranınız **{fat_percentage:.2f}%** ve tahmin edilen kalori yakımınız **{predicted_calories:.2f} kalori** olarak hesaplanmıştır.\n\n"
                    "Sonuçlarınız, kilonuzun biraz düşük olduğunu ve daha dengeli bir diyet ile kilo almayı hedeflemeniz gerektiğini göstermektedir. \n"
                    "Daha fazla kas kütlesi oluşturmak için protein ağırlıklı beslenme planlarını ve düzenli egzersizleri değerlendirmeniz önerilir."
                )
            elif 18.5 <= bmi < 24.9:
                st.success(
                    f"Tebrikler! Yaptığınız antrenmanın etkisini ölçerken, **BMI** değeriniz **{bmi:.2f}**, yağ oranınız **{fat_percentage:.2f}%** ve yakılan tahmini kalori miktarı **{predicted_calories:.2f} kalori** olarak belirlenmiştir.\n\n"
                    "Bu değerler, ideal sağlık aralığında olduğunuzu göstermektedir. Mevcut rutininizi koruyarak, genel sağlığınızı ve fitness düzeyinizi sürdürebilirsiniz."
                )
            elif 25 <= bmi < 29.9:
                st.warning(
                    f"Sağlık açısından daha iyi bir denge yakalamak için BMI değeriniz **{bmi:.2f}**, yağ oranınız **{fat_percentage:.2f}%** ve tahmin edilen kalori yakımınız **{predicted_calories:.2f} kalori** olarak hesaplanmıştır.\n\n"
                    "Bu sonuçlar, biraz fazla kiloya sahip olduğunuzu gösterebilir. Daha fazla kalori yakımını desteklemek ve ideal BMI aralığına ulaşmak için egzersiz yoğunluğunuzu artırmayı ve sağlıklı bir diyet planlamayı düşünebilirsiniz."
                )
            else:
                st.error(
                    f"Bugünkü egzersizinizle ilgili sonuçlarınızı özetlemek gerekirse: **BMI** değeriniz **{bmi:.2f}**, vücut yağ oranınız **{fat_percentage:.2f}%** ve yakılan tahmini kalori miktarı **{predicted_calories:.2f} kalori** olarak hesaplanmıştır.\n\n"
                    "Bu sonuçlar, obezite kategorisinde yer aldığınızı gösterebilir. Sağlıklı bir kilo kontrolü için bir beslenme uzmanı veya fitness uzmanı ile çalışmanız, düzenli fiziksel aktivite ve sağlıklı bir diyet planı oluşturmanız önerilir."
                )

            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_calories,
                title={'text': "Tahmini Kalori Yakımı"},
                gauge={
                    'axis': {'range': [0, 1500]},
                    'steps': [
                        {'range': [0, 500], 'color': "lightgray"},
                        {'range': [500, 1000], 'color': "lightgreen"},
                        {'range': [1000, 1500], 'color': "red"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': predicted_calories}
                }
            ))

            
            categories = ['BMI', 'Fat %', 'Exercise Duration (hr)', 'Water Intake (L)']
            user_data = [bmi, fat_percentage, exercise_duration / 60, water_intake]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=user_data,
                theta=categories,
                fill='toself',
                name='Kullanıcı'
            ))

            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                st.plotly_chart(fig_radar, use_container_width=True)

        else:
            st.warning("Lütfen cinsiyet ve egzersiz türünü seçin.")
