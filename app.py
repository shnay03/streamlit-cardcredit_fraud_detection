import streamlit as st
import pandas as pd  # Add this line
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Membaca model SVM dari berkas pickle
with open('svm_model.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Fungsi untuk melakukan prediksi menggunakan model SVM
def predict_fraud(age, address_duration, device_mobile, consistency_data, location_unusual, employment_verification):
    # Memasukkan input ke dalam dataframe
    input_data = {'Age': [age],
                  'Address Duration': [address_duration],
                  'Device Information Mobile': [device_mobile],
                  'Consistency in Data Inconsistent': [consistency_data],
                  'Location of Application Unusual': [location_unusual],
                  'Employment Verification Not Verified': [employment_verification]}
    
    input_df = pd.DataFrame(input_data)
    
    # Melakukan prediksi menggunakan model SVM
    prediction = svm_model.predict(input_df)
    
    return prediction[0]

# Membuat aplikasi Streamlit
st.write('# Card Credit Fraud Detection')

# Input dari pengguna
age = st.slider('Age', min_value=18, max_value=100, value=25)
address_duration = st.slider('Address Duration (years)', min_value=0, max_value=50, value=5)
device_mobile = st.checkbox('Device Information Mobile')
consistency_data = st.checkbox('Consistency in Data Inconsistent')
location_unusual = st.checkbox('Location of Application Unusual')
employment_verification = st.checkbox('Employment Verification Not Verified')

# Tombol untuk melakukan prediksi
if st.button('Predict'):
    result = predict_fraud(age, address_duration, device_mobile, consistency_data, location_unusual, employment_verification)
    st.write('### Prediction Result:')
    if result == 1:
        st.write('This data may be fraud!')
    else:
        st.write('This data is not detected as fraud.')
