# Telcom Customer Churn Prediction App

Welcome to our Telcom Customer Churn Prediction App! This application is designed to help you predict whether a customer will churn or not based on various features. By analyzing customer data such as contract type, payment method, and service usage, the model provides valuable insights to help you make informed decisions and enhance customer retention strategies.

## How to Use the App

1. **Input Fields**: Fill in the details in the form below. You'll need to provide information related to the customer's profile, such as whether they are a senior citizen, their partner status, internet service type, and more.

2. **Predict Button**: Click the 'Predict' button to see whether the customer is likely to churn. The app will analyze the input data and provide a prediction.

## Input Features

The following features are used for prediction:

- **Senior Citizen**: Select 'Yes' or 'No'.
- **Partner**: Select 'Yes' or 'No'.
- **Dependents**: Select 'Yes' or 'No'.
- **Multiple Lines**: Select 'Yes' or 'No'.
- **Internet Service**: Choose from 'DSL', 'Fiber optic', or 'No'.
- **Online Security**: Select 'Yes' or 'No'.
- **Online Backup**: Select 'Yes' or 'No'.
- **Device Protection**: Select 'Yes' or 'No'.
- **Tech Support**: Select 'Yes' or 'No'.
- **Streaming TV**: Select 'Yes' or 'No'.
- **Streaming Movies**: Select 'Yes' or 'No'.
- **Contract**: Choose from 'Month-to-month', 'One year', or 'Two year'.
- **Paperless Billing**: Select 'Yes' or 'No'.
- **Payment Method**: Choose from 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', or 'Credit card (automatic)'.
- **Monthly Charges**: Enter the monthly charges (numeric value).
- **Total Charges**: Enter the total charges (numeric value).
- **Tenure Group**: Choose from '0-12', '13-24', '25-36', '37-48', '49-60', or '61-72'.

## Prediction Result

After clicking the 'Predict' button, the app will display whether the customer is likely to churn or not.

Feel free to explore and use the app to make data-driven decisions!

---

*Note: If you encounter any errors, please check your input values or contact the app administrator.*

## How to Run the App Locally

1. Clone this repository to your local machine.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Run the app using `streamlit run app.py`.

---

*For more information, visit the Streamlit documentation.*

## Acknowledgments

This app was created as part of a machine learning project. We acknowledge the contributions of the following resources:

- Streamlit
- Scikit-learn
- Pandas
- Joblib


