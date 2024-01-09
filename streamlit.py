import streamlit as st
import pandas as pd
import pickle

# Load the pickled logistic regression model
model = pickle.load(open('logistic_reg.pkl', 'rb'))

# Create a Streamlit web app
st.title("Diabetes Prediction App")

# Add a brief description
st.write("This app predicts whether a person has diabetes or not based on input features.")

# Create input fields for user to enter data
st.sidebar.header("User Input Features")

# Input fields for user data
pregnancies = st.sidebar.number_input("Pregnancies", 0, 17, 3)
glucose = st.sidebar.number_input("Glucose", 0, 199, 117)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 122, 72)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 99, 23)
insulin = st.sidebar.number_input("Insulin", 0, 846, 30)
bmi = st.sidebar.number_input("BMI", 0.0, 67.1, 32.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
age = st.sidebar.number_input("Age", 21, 81, 29)

# Create a "Predict" button
predict_button = st.sidebar.button("Predict")

# Check if the "Predict" button is clicked
if predict_button:
    # Create a DataFrame from user input
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    # Make predictions
    prediction = model.predict(user_data)
    prediction_proba = model.predict_proba(user_data)[:, 1]

    # Display prediction results
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("The model predicts that the person has diabetes.")
        st.write(f"Probability of having diabetes: {prediction_proba[0]:.2f}")
    else:
        st.write("The model predicts that the person does not have diabetes.")
        st.write(f"Probability of not having diabetes: {1 - prediction_proba[0]:.2f}")
