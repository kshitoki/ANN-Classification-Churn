import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#load the model
model = tf.keras.models.load_model('ann_model.h5')

#load all the pickle files
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encode_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('one_hot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)


#streamlit app
st.title("Customer Churn Prediction")


#User input data
geogrpahy = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
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
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geogrpahy]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

#prediction

pridiction = model.predict(input_data_scaled)
prediction_prob = pridiction[0][0]

st.write(f"Prediction Probability: {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")


