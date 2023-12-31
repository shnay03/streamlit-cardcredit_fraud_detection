# -*- coding: utf-8 -*-
"""fraud_streamlit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16eRkyvUYAF1eaxxVx8Fhv9IETKF444me
"""

!pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler

# Load Dataset

account_data = pd.read_csv('https://github.com/csry15/Fraud_data/blob/main/account_data.csv/?raw=True')

account_data['IsFraud'] = account_data['IsFraud'].replace({'Yes': 1, 'No': 0})

data_temp = pd.get_dummies(account_data[['Occupation', 'MaritalStatus','ResidentialStatus', 'PurposeoftheLoan', 'Collateral', 'ApplicationBehavior',
                                         'LocationofApplication', 'ChangeinBehavior', 'AccountActivity', 'PaymentBehavior', 'Blacklists',
                                         'EmploymentVerification', 'PastFinancialMalpractices', 'DeviceInformation', 'SocialMediaFootprint',
                                         'ConsistencyinData', 'Referral']])
account_df = pd.concat([account_data, data_temp], axis=1)
account_df.drop(columns=['Occupation', 'MaritalStatus', 'ResidentialStatus', 'PurposeoftheLoan',  'Collateral', 'ApplicationBehavior', 'LocationofApplication',
                         'ChangeinBehavior', 'AccountActivity', 'PaymentBehavior', 'Blacklists', 'EmploymentVerification', 'PastFinancialMalpractices',
                         'DeviceInformation', 'SocialMediaFootprint', 'ConsistencyinData', 'Referral'], inplace=True)

account_df = account_df.loc[:, ~account_df.columns.str.contains('^TimeofTransaction')]

# Variabel X dan y seperti yang Anda berikan
y = account_df['IsFraud']
X = account_df.drop(['Age','AddressDuration', 'DeviceInformation_Mobile',
                     'ConsistencyinData_Inconsistent', 'LocationofApplication_Unusual',
                     'EmploymentVerification_Not Verified'], axis=1)

# Train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=15, stratify=y)

# Train the SVM model
svm_model = SVC(C=1.0, kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

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

# Streamlit app
st.title("Credit Card Fraud Prediction")

# User-friendly layout and styling
st.sidebar.title("User Input")
st.sidebar.markdown("Please provide the following information:")

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