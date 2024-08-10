import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load the pre-trained model and preprocessor
model = joblib.load('gradient_boosting_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Define the input fields for the 16 original parameters
def user_input_features():
    col1, col2, col3 = st.columns(3)

    with col1:
        Senior_Citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
        Partner = st.selectbox('Partner', ['Yes', 'No'])
        Dependents = st.selectbox('Dependents', ['Yes', 'No'])
        Multiple_Lines = st.selectbox('Multiple Lines', ['Yes', 'No'])
        Internet_Service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

    with col2:
        Online_Security = st.selectbox('Online Security', ['Yes', 'No'])
        Online_Backup = st.selectbox('Online Backup', ['Yes', 'No'])
        Device_Protection = st.selectbox('Device Protection', ['Yes', 'No'])
        Tech_Support = st.selectbox('Tech Support', ['Yes', 'No'])
        Streaming_TV = st.selectbox('Streaming TV', ['Yes', 'No'])

    with col3:
        Streaming_Movies = st.selectbox('Streaming Movies', ['Yes', 'No'])
        Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        Paperless_Billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        Payment_Method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        Monthly_Charges = st.number_input('Monthly Charges', min_value=0.0)
        Total_Charges = st.number_input('Total Charges', min_value=0.0)
        tenure_group = st.selectbox('Tenure Group', ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])

    data = {
        'Senior_Citizen': Senior_Citizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'Multiple_Lines': Multiple_Lines,
        'Internet_Service': Internet_Service,
        'Online_Security': Online_Security,
        'Online_Backup': Online_Backup,
        'Device_Protection': Device_Protection,
        'Tech_Support': Tech_Support,
        'Streaming_TV': Streaming_TV,
        'Streaming_Movies': Streaming_Movies,
        'Contract': Contract,
        'Paperless_Billing': Paperless_Billing,
        'Payment_Method': Payment_Method,
        'Monthly_Charges': Monthly_Charges,
        'Total_Charges': Total_Charges,
        'tenure_group': tenure_group
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Main app function
def main():
    # Streamlit app configuration
    st.set_page_config(page_title="Telcom Customer Churn Prediction App", layout="centered")

    # Apply custom CSS and theme
    st.markdown(
        """
        <style>
        .big-font {
            font-size:30px !important;
            color: #333;
        }
        .result {
            font-size:50px !important;
            font-weight: bold;
            color: #DA7297; /* Tomato color for prediction result */
            text-align: center;
        }
        .write-up {
            font-size: 18px;
            color: #1A3636;
        }
        .divider {
            border-top: 2px solid #FFB4C2;
            margin: 20px 0;
        }
        .spacer {
            height: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Page Title
    st.title("Telcom Customer Churn Prediction App")

    # Hero section with text and image
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(
            """
            <div class="write-up">
            Welcome to our Telcom Customer Churn Prediction App! This application is designed to help you predict whether a customer will churn or not based on various features. By analyzing customer data such as contract type, payment method, and service usage, the model provides valuable insights to help you make informed decisions and enhance customer retention strategies.
            <br><br>
            Simply fill in the details in the form below and click the 'Predict' button to see whether the customer is likely to churn. 
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.image("heroimage.png", caption="Churn Prediction", use_column_width=True)

    # Divider between the introduction and input fields
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Space for better view
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # Get user input
    input_df = user_input_features()

    # Add a button to make predictions
    if st.button('Predict'):
        try:
            # Preprocess the user input using the trained preprocessor
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_transformed)
            
            # Display the prediction result
            result = "Churn" if prediction[0] == 1 else "No Churn"
            st.markdown(f"<p class='result'>Prediction: {result}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
