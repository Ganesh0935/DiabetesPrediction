import streamlit as st
import pickle
import numpy as np
import pandas as pd

def main():
    # Set page layout to wide to have more horizontal space
    st.set_page_config(page_title="Diabetes Prediction", layout="wide")
    # Inject custom CSS to style margins, remove spinners, and make the button full width
    st.markdown("""
    <style>
    /* Center the main block and add padding for a cleaner look */
    .main {
        max-width: 900px;
        margin: 100px auto;
        padding: 2rem;
        background-color: #f9f9f9;
        border-radius: 8px;
    }
    /* Center columns horizontally */
    [data-testid="stHorizontalBlock"] {
        justify-content: center;
    }
    /* Remove up/down arrows (spinners) from number input fields in some browsers */
    [data-baseweb="input"] input[type=number]::-webkit-inner-spin-button,
    [data-baseweb="input"] input[type=number]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    /* Make the button full width */
    .full-width-button > div.stButton {
        width: 100%;
    }
    .full-width-button > div.stButton > button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
        background-color: #4CAF50;
        color: white;
        border-radius: 6px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Diabetes Prediction using ML")
    # Load your trained pipeline from the pickle file
    with open('diabetes_pipeline.pkl', 'rb') as file:
        diabetes_pipeline = pickle.load(file)
    st.subheader("Enter Patient Details:")
    # First row (3 fields)
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        glucose = st.text_input("Glucose Level")
    with col3:
        bp = st.text_input("Blood Pressure Level")
    # Second row (3 fields)
    col4, col5, col6 = st.columns(3)
    with col4:
        skin_thickness = st.text_input("Skin Thickness")
    with col5:
        insulin = st.text_input("Insulin Level")
    with col6:
        bmi = st.text_input("BMI Value")
    # Third row (2 fields)
    col7, col8 = st.columns(2)
    with col7:
        dpf = st.text_input("Diabetes Pedigree Function Value")
    with col8:
        age = st.text_input("Age of the Person")
    
    # Full-width button container
    st.markdown("<div class='full-width-button'>", unsafe_allow_html=True)
    if st.button("Predict Diabetes Status"):
        try:
            # Convert inputs to float
            pregnancies_val = float(pregnancies)
            glucose_val = float(glucose)
            bp_val = float(bp)
            skin_thickness_val = float(skin_thickness)
            insulin_val = float(insulin)
            bmi_val = float(bmi)
            dpf_val = float(dpf)
            age_val = float(age)
            
            # Create input data dictionary
            input_data = {
                'Pregnancies': pregnancies_val,
                'Glucose': glucose_val,
                'BloodPressure': bp_val,
                'SkinThickness': skin_thickness_val,
                'Insulin': insulin_val,
                'BMI': bmi_val,
                'DiabetesPedigreeFunction': dpf_val,
                'Age': age_val
            }
            
            # Create DataFrame from the dictionary
            input_df = pd.DataFrame([input_data])
            
            # Add the one-hot encoded age features based on the input age
            input_df['Age_<30'] = 1 if age_val < 30 else 0
            input_df['Age_30-40'] = 1 if 30 <= age_val < 40 else 0
            input_df['Age_40-50'] = 1 if 40 <= age_val < 50 else 0
            input_df['Age_>50'] = 1 if age_val >= 50 else 0
            
            # Predict using the loaded pipeline
            prediction = diabetes_pipeline.predict(input_df)
            
            # Display results with a bit of spacing
            st.markdown("<br>", unsafe_allow_html=True)
            if prediction[0] == 1:
                st.error("The patient is likely to have diabetes.")
            else:
                st.success("The patient is likely NOT to have diabetes.")
                
        except ValueError:
            st.warning("Please enter valid numeric values in all fields.")
        except Exception as e:
            st.warning(f"An error occurred: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
