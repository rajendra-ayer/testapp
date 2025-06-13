import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

st.title(" Train Linear Regression Model")

uploaded_file = st.file_uploader("Upload CSV datasets'", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of data:")
        st.write(df.head())


        model = LinearRegression()
        model.fit(df[['Area']], df['Price'])

        with open("model.pkl", "wb") as f:
             pickle.dump(model, f)

        st.success("✅ Model trained and saved as model.pkl")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a CSV file to train the model.")
