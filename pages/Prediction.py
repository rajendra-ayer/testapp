import streamlit as st
import pandas as pd
from skops.io import load

# Title of page
st.title("House price prediction")

# Load model file
model=load("model/model.skops")
st.write("Model loaded successfully")

# User data
user_input=pd.DataFrame({
    "Area_sqft":[st.number_input("Put Area:")],
    "Bedrooms":[st.number_input("Put Number of Bedrooms:")],
    "Age_years":[st.number_input("Put age of the property:")]
})

# Prediction
prediction=model.predict(user_input)
if st.button("Predict"):
    st.write(f'''The predicted house price is" $ 
             
             {prediction[0]}''')