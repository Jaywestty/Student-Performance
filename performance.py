# Import Necessary Libraries
import streamlit as st
import pandas as pd
import pickle

# Load the saved model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Page Config
st.set_page_config(page_title="Student Performance App", page_icon="🎓", layout="centered")

# Custom CSS for light theme and better UI
st.markdown("""
    <style>
        .main {
            background-color: #f9fafa;
            padding: 2rem;
            border-radius: 10px;
        }
        h1, h2, h3 {
            color: #2b547e;
        }
        .stButton>button {
            background-color: #2b547e;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #1f3d5a;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Description
st.markdown("<h1 style='text-align: center;'>🎓 Student Performance App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Fill in student habits and background details to predict performance.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Fields (Single-column Layout)
age = st.number_input('Age', min_value=10, max_value=60, value=20)
study_hours_per_day = st.slider('📚 Study Hours Per Day', 0.0, 12.0, 3.0)
social_media_hours = st.slider('📱 Hours on Social Media', 0.0, 12.0, 3.0)
netflix_hours = st.slider('🎬 Hours on Netflix', 0.0, 12.0, 3.0)
attendance_percentage = st.slider('🏫 Attendance Rate (%)', 0.0, 100.0, 60.0)
sleep_hours = st.slider('🛌 Sleep Hours per Day', 0.0, 24.0, 7.0)
exercise_frequency = st.slider('💪 Exercise Frequency (per week)', 0, 7, 3)
mental_health_rating = st.slider('🧠 Mental Health Rating (1 - 10)', 1, 10, 6)

gender = st.selectbox('👤 Gender', ['Male', 'Female'])
part_time_job = st.selectbox('💼 Part-time Job', ['Yes', 'No'])
diet_quality = st.selectbox('🥗 Diet Quality', ['Poor', 'Average', 'Good', 'Excellent'])
internet_quality = st.selectbox('🌐 Internet Quality', ['Poor', 'Average', 'Good', 'Great'])
extracurricular_participation = st.selectbox('🎯 Curricular Activities', ['Yes', 'No'])
parental_education_level = st.selectbox('🎓 Parent Education Level', ['None', 'High School', 'Bachelor', 'Master'])

st.markdown("---")

# Prediction Button
if st.button('🚀 Predict Performance'):
    input_data = {
        'age': age,
        'study_hours_per_day': study_hours_per_day,
        'social_media_hours': social_media_hours,
        'netflix_hours': netflix_hours,
        'attendance_percentage': attendance_percentage,
        'sleep_hours': sleep_hours,
        'exercise_frequency': exercise_frequency,
        'mental_health_rating': mental_health_rating,
        'gender': gender,
        'part_time_job': part_time_job,
        'diet_quality': diet_quality,
        'internet_quality': internet_quality,
        'extracurricular_participation': extracurricular_participation,
        'parental_education_level': parental_education_level
    }

    input_df = pd.DataFrame([input_data])

    # Predict
    prediction = model.predict(input_df)[0]

    # Display Result
    st.success(f"🎯 **Predicted Student Performance:** `{prediction}`")
