import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="BurnedCalories", page_icon="🤸‍♂️")

st.title(":orange[Your]:blue[ Burned Calories]:red[ On Workout]")

@st.cache
def load_data():
    return pd.read_csv("dataset/lastDataset.csv")

data = load_data()

X = data.drop(columns=["Calories_Burned"])
y = data["Calories_Burned"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

le_gender = LabelEncoder()
le_workout_type = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])
X["Workout_Type"] = le_workout_type.fit_transform(X["Workout_Type"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

st.header("Kişisel Bilgileriniz")

age = st.slider("Yaşınızı Seçin", min_value=18, max_value=80, value=25, step=1)

st.markdown("Cinsiyetinizi Seçin")
col1, col2 = st.columns(2)

if "gender" not in st.session_state:
    st.session_state["gender"] = None
    st.session_state["gender_value"] = None

with col1:
    if st.button("Kadın", key="woman_button"):
        st.session_state["gender"] = 0
        st.session_state["gender_value"] = 0
    st.image("images/woman.png", width=200, caption="Kadın")

with col2:
    if st.button("Erkek", key="man_button"):
        st.session_state["gender"] = 1
        st.session_state["gender_value"] = 1
    st.image("images/man.png", width=200, caption="Erkek")

gender = st.session_state["gender"]
gender_value = st.session_state["gender_value"]

height = st.slider("Boyunuzu Seçin (cm)", min_value=120, max_value=230, value=170, step=1)
weight = st.slider("Kilonuzu Seçin (kg)", min_value=40, max_value=150, value=70, step=1)

st.subheader("Kalp Ritmi (Nabız)")
max_bpm = st.number_input("Maksimum Nabız", min_value=0, step=1)
avg_bpm = st.number_input("Ortalama Nabız", min_value=0, step=1)
resting_bpm = st.number_input("Dinlenme Nabzı", min_value=0, step=1)

exercise_duration = st.slider("Egzersiz Süresi (dakika)", min_value=0, max_value=300, value=60, step=1)
exercise_duration_hours = exercise_duration / 60

st.subheader("Egzersiz Türünüzü Seçin")
col1, col2, col3, col4 = st.columns(4)

if "exercise_type" not in st.session_state:
    st.session_state["exercise_type"] = None

with col1:
    if st.button("Cardio", key="cardio_button"):
        st.session_state["exercise_type"] = 0
    st.image("images/cardio.png", width=200, caption="Cardio")

with col2:
    if st.button("HIIT", key="hiit_button"):
        st.session_state["exercise_type"] = 1
    st.image("images/hiit.png", width=200, caption="HIIT")

with col3:
    if st.button("Strength", key="strength_button"):
        st.session_state["exercise_type"] = 2
    st.image("images/strength.png", width=200, caption="Strength")

with col4:
    if st.button("Yoga", key="yoga_button"):
        st.session_state["exercise_type"] = 3
    st.image("images/yoga.png", width=200, caption="Yoga")

exercise_type = st.session_state["exercise_type"]
workout_type_value = le_workout_type.transform([exercise_type])[0] if exercise_type else None

water_intake = st.slider("Su İçtiyseniz Kaç Litre?", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

height_m = height / 100
bmi = weight / (height_m ** 2)
fat_percentage = 1.20 * bmi + 0.23 * age - (16.2 if gender == "Erkek" else 5.4)

if st.button("Tahminleme Yap"):
    if gender_value is not None and workout_type_value is not None:
        input_data = np.array([[age, gender_value, weight, height_m, max_bpm, avg_bpm, resting_bpm, exercise_duration_hours, workout_type_value, fat_percentage, water_intake, bmi]])
        
        input_data = np.array([[age, gender_value, weight, height_m, max_bpm, avg_bpm, resting_bpm, exercise_duration_hours, workout_type_value, fat_percentage, water_intake, bmi]])
        input_data_scaled = scaler.transform(input_data)

        predicted_calories = model.predict(input_data_scaled)[0]
        
        st.subheader("Tahmin Sonucu")
        st.write(f"Yaş: {age} yıl")
        st.write(f"Cinsiyet: {gender}")
        st.write(f"Boy: {height} cm")
        st.write(f"Kilo: {weight} kg")
        st.write(f"BMI: {bmi:.2f}")
        st.write(f"Yağ Oranı: {fat_percentage:.2f}%")
        st.write(f"Egzersiz Türü: {exercise_type}")
        st.write(f"Egzersiz Süresi: {exercise_duration} dakika ({exercise_duration_hours:.2f} saat)")
        st.write(f"Su Tüketimi: {water_intake} litre")
        st.write(f"Tahmin Edilen Yakılan Kalori: {predicted_calories:.2f} kalori")


        # Kullanıcıya gösterilecek mesajlar
        if bmi < 18.5:
            st.success(f"Egzersiz verileriniz incelendiğinde, BMI değeriniz {bmi:.2f}, yağ oranınız {fat_percentage:.2f}% ve tahmin edilen kalori yakımınız {predicted_calories:.2f} kalori olarak hesaplanmıştır. Düzenli egzersiz ve yeterli beslenme, kilo alımını destekleyebilir.")
        elif 18.5 <= bmi < 24.9:
            st.success(f"Yaptığınız antrenmanın etkisini ölçerken, BMI'niz {bmi:.2f}, yağ oranınız {fat_percentage:.2f}% ve yakılan tahmini kalori miktarı {predicted_calories:.2f} kalori olarak belirlenmiştir. Bu değerler, ideal aralıktadır; mevcut rutininizi korumanız önerilir.")
        elif 25 <= bmi < 29.9:
            st.warning(f"Sağlığınızı ve fitness hedeflerinizi daha iyi takip etmek için BMI değerinizin {bmi:.2f}, yağ oranınızın {fat_percentage:.2f}% olduğunu ve egzersiziniz sırasında tahmini olarak {predicted_calories:.2f} kalori yaktığınızı hesapladık. Daha fazla kalori yakımını desteklemek için düzenli egzersize devam edebilirsiniz.")
        else:
            st.error(f"Bugünkü egzersizinizle ilgili sonuçlarınızı özetlemek gerekirse: BMI değeriniz {bmi:.2f}, vücut yağ oranınız {fat_percentage:.2f}% ve yakılan tahmini kalori {predicted_calories:.2f} kalori olarak hesaplanmıştır. Sağlıklı bir kilo kontrolü için beslenme ve egzersiz planınızı gözden geçirmeniz önerilir.")

    else:
        st.warning("Lütfen cinsiyet ve egzersiz türünü seçin.")