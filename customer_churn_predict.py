import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import plotly.graph_objects as go
from math import pi
from sklearn.metrics import confusion_matrix

# Load the pre-trained model and preprocessor
model = joblib.load('prediction_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

import streamlit as st
import pandas as pd

def user_input_features():
    # Demographics Section
    st.header('Demographics')
    col1, col2, col3 = st.columns(3)
    with col1:
        Senior_Citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
    with col2:
        Gender = st.selectbox('Gender', ['Male', 'Female'])
    with col3:
        Partner = st.selectbox('Partner', ['Yes', 'No'])
    with col1:  # Continuing in the first column
        Dependents = st.selectbox('Dependents', ['Yes', 'No'])

    # Location Section
    st.header('Location')
    col1, col2, col3 = st.columns(3)
    with col1:
        Zip_Code = st.number_input('Zip Code', min_value=90001, max_value=96161, value=96150)

    # Services Section
    st.header('Services')
    col1, col2, col3 = st.columns(3)
    with col1:
        Phone_Service = st.selectbox('Phone Service', ['Yes', 'No'])
        Online_Security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
        Device_Protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
        Streaming_TV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
        Payment_Method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

    with col2:
        Multiple_Lines = st.selectbox('Multiple Lines', ['Yes', 'No'])
        Online_Backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
        Tech_Support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
        Streaming_Movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
        Monthly_Charges = st.number_input('Monthly Charges', min_value=0.0)

    with col3:
        Internet_Service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        Paperless_Billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
        Total_Charges = st.number_input('Total Charges', min_value=0.0)
        tenure_group = st.selectbox('Tenure Group', ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])

    # Collecting data in a dictionary
    data = {
        'Zip_Code': Zip_Code,
        'Gender': Gender,
        'Senior_Citizen': Senior_Citizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'Multiple_Lines': Multiple_Lines,
        'Phone_Service' : Phone_Service,
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

    # Convert the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

def display_prediction(prediction, probability=None, input_df=None, model=None):
    # Load images
    churn_image = Image.open('unhappy.jpg')  # Replace with your churn image path
    no_churn_image = Image.open('happy.jpg')  # Replace with your no churn image path

    # Determine the result
    result = "Churn" if prediction == 1 else "No Churn"
    st.markdown(f"<p class='result'>Prediction: {result}</p>", unsafe_allow_html=True)
    
    # Display prediction result in a well-formatted manner
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>Customer Churn Prediction</h2>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        if prediction == 1:
            st.image(churn_image, caption='Customer is likely to churn!', use_column_width=True)
        else:
            st.image(no_churn_image, caption='Customer is likely to stay!', use_column_width=True)

    st.markdown("---")

    if probability is not None:
        st.markdown("<h3 style='text-align: center; color: #4B8BFF;'>Prediction Probability</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 18px;'>There is a <strong>{probability * 100:.2f}%</strong> probability of churn.</p>", unsafe_allow_html=True)
        
        # Create a colorful pie chart
        fig_pie = go.Figure(data=[go.Pie(labels=['No Churn', 'Churn'], 
                                         values=[1-probability, probability],
                                         textinfo='label+percent', 
                                         insidetextorientation='radial',
                                         marker=dict(colors=['#EECEB9', '#987D9A']))])
        fig_pie.update_traces(hoverinfo='label+percent+value', textfont_size=15)
        st.plotly_chart(fig_pie)

        # Create a colorful bar chart
        # st.markdown("<h3 style='text-align: center; color: #4B8BFF;'>Probability Distribution</h3>", unsafe_allow_html=True)
        # fig, ax = plt.subplots()
        # sns.barplot(x=['No Churn', 'Churn'], y=[1-probability, probability], palette="coolwarm", ax=ax)
        # ax.set_ylabel('Probability')
        # ax.set_xlabel('')
        # ax.set_title('Probability of Churn')
        # st.pyplot(fig)

        # st.markdown("---")
        


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
    # Print the features dataframe
    with pd.option_context('display.max_columns', None):
        print("Features:")
        print(input_df)
    
    # Divider between the introduction and input fields
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Add a button to make predictions
    if st.button('Predict'):
        try:
            # Preprocess the user input using the trained preprocessor
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction and get probability
            prediction = model.predict(input_transformed)
            prediction_proba = model.predict_proba(input_transformed)[0, 1]  # Probability of churn
            
            # Display the prediction result with visualization
            display_prediction(prediction[0], prediction_proba,input_df=input_df, model=model)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Team Information
    st.markdown("""
    <style>
        .divider {
            height: 3px;
            background-color: #FF4B4B;
            margin: 40px 0;
        }

        .footer {
            padding: 20px;
            background-color: #F8EDE3;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            text-align: center;
        }

        .team-title {
            font-size: 28px;
            font-weight: bold;
            color: #FF4B4B;
            margin-bottom: 20px;
        }

        .team-member {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin-bottom: 20px;
            background-color: #F8EDE3;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .team-member img {
            border-radius: 50%;
            width: 80px;
            height: 80px;
            margin-right: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .team-info {
            text-align: left;
        }

        .team-info h4 {
            margin: 0;
            font-size: 20px;
            color: #333;
        }

        .team-info p {
            margin: 5px 0 0;
            font-size: 14px;
            color: #666;
        }

        .role {
            font-size: 16px;
            font-weight: bold;
            color: #007BFF;
        }

        .social-icons {
            margin-top: 10px;
        }

        .social-icons img {
            width: 24px;
            height: 24px;
            margin-right: 10px;
        }

        .team-section {
            display: flex;
            flex-direction: column;
            align-items: left;
        }
    </style>
    <div class="divider"></div>
    <div class="footer">
        <div class="team-title">ITI105 - Group 11 - Team members </div>
        <div class="team-section">
            <div class="team-member">
                <img src="https://via.placeholder.com/80" alt="Manikandan Sadhasivam">
                <div class="team-info">
                    <h4>Manikandan Sadhasivam</h4>
                </div>
            </div>
            <div class="team-member">
                <img src="https://via.placeholder.com/80" alt="Anand Geetha">
                <div class="team-info">
                    <h4>Anand Geetha</h4>
                </div>
            </div>
            <div class="team-member">
                <img src="https://via.placeholder.com/80" alt="Angappan Sathyabama">
                <div class="team-info">
                    <h4>Angappan Sathyabama</h4>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
