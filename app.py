import streamlit as st  # Importing Streamlit library for creating web app interface
import numpy as np  # Importing NumPy for numerical operations
import pandas as pd  # Importing Pandas for data manipulation
import pickle  # Importing pickle for loading the machine learning model

# Load the pre-trained Random Forest Regression model
rfr = pickle.load(open('model.pkl', 'rb'))

def pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    """
    Function to make predictions using the loaded machine learning model.

    Parameters:
    Gender (int): 0 for Male, 1 for Female
    Age (int): Age of the person
    Height (int): Height in centimeters
    Weight (int): Weight in kilograms
    Duration (int): Duration of exercise in minutes
    Heart_rate (int): Heart rate in beats per minute
    Body_temp (float): Body temperature in degrees Celsius

    Returns:
    float: Predicted calories burned based on input features
    """
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])  # Create feature array
    prediction = rfr.predict(features)  # Make prediction
    return prediction[0]  # Return the predicted value

# Custom CSS styling for the Streamlit app interface
st.markdown("""
    <style>
        body {
            background-color: #f0f0f0;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            color: #FFFFFF;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .form-container {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            border: 2px solid #1F77B4;
            max-width: 600px;
            margin: auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stSelectbox, .stButton {
            margin-top: 1rem;
        }
        .stButton button {
            background-color: #1F77B4;
            color: white;
            border-radius: 5px;
            display: block;
            margin: auto;
        }
        .prediction {
            color: yellow;
            font-size: 1.5rem;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Setting the title of the Streamlit app
st.markdown("<h1 class='main-title'>Calories Burn Prediction</h1>", unsafe_allow_html=True)

# Define options and ranges for input fields
gender_options = {"Male": 0, "Female": 1}
age_range = list(range(20, 81))
height_range = list(range(36, 223))
weight_range = list(range(26, 123))
duration_range = list(range(1, 61))
heart_rate_range = list(range(67, 129))
body_temp_range = [round(x, 1) for x in np.arange(37.0, 45.0, 0.1)]

# Creating input fields for user input
Gender = st.selectbox('Gender', options=list(gender_options.keys()))

col1, col2 = st.columns(2)
with col1:
    Age = st.selectbox('Age', options=age_range)
with col2:
    Height = st.selectbox('Height', options=height_range)

col3, col4 = st.columns(2)
with col3:
    Weight = st.selectbox('Weight', options=weight_range)
with col4:
    Duration = st.selectbox('Duration (minutes)', options=duration_range)

col5, col6 = st.columns(2)
with col5:
    Heart_rate = st.selectbox('Heart Rate (bpm)', options=heart_rate_range)
with col6:
    Body_temp = st.selectbox('Body Temperature (°C)', options=body_temp_range)

# Making prediction upon button click
if st.button('Predict'):
    Gender_value = gender_options[Gender]  # Convert gender to numerical value
    result = pred(Gender_value, Age, Height, Weight, Duration, Heart_rate, Body_temp)  # Get prediction
    # Display the prediction result
    st.markdown(f"<div class='prediction'>You have consumed this amount of calories: {result:.2f}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # Closing div tag for styling
