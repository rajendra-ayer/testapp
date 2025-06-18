import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from skops.io import dump


# App Title
st.title("AI APP TRAINING")

# Load Models
df=pd.read_csv("housingdata.csv")
st.write(df.head())

X=df[['Area_sqft', 'Bedrooms', 'Age_years']]
y=df['Price_$']
model=LinearRegression()
model_training=model.fit(X,y)

if st.button("Train Model"):
    st.write("Model trained and saved successfully as model.skops")
    dump(model_training, "model/model.skops")