import keras
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#load the model
model = keras.models.load_model('salary_regression_model.h5')

#load all the pickle files
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encode_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)


#streamlit app
st.title("Estimate salary prediction")


#User input data
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])


#creating a dataframe for the input data
input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Exited' : [exited],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member]
})

geo_df = pd.DataFrame({'Geography': [geography]})
geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data = input_data.reindex(columns=columns, fill_value=0)

st.write(input_data.columns.tolist())
    
input_data_scaled = scaler.transform(input_data)

st.write("Final columns used:", input_data.columns.tolist())
st.write("Expected columns:", list(columns))

#prediction

prediction = model.predict(input_data_scaled)
prediction_salary= prediction[0][0]

st.write(f"Predicted Salary: Rs{prediction_salary:.2f}")

