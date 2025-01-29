import streamlit as st
import pandas as pd
from model_trainer import ModelTrainer
from data_processor import DataProcessor
from visualizer import plot_model_comparisons, plot_feature_importance

def main():
    st.title("Depression Screening Application")
    
    # Initialize processors
    data_processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Sidebar for data upload
    st.sidebar.header("Training Data")
    uploaded_file = st.sidebar.file_uploader("Upload training dataset", type=["csv"])
    
    if uploaded_file is not None:
        # Load and preprocess training data
        train_data = pd.read_csv(uploaded_file)
        processed_train_data = data_processor.preprocess_data(train_data)
        
        # Main input form
        st.header("Please Answer the Following Questions")
        
        # Create input form
        with st.form("screening_form"):
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
            study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
            
            sleep_duration = st.selectbox("Sleep Duration", 
                ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
            
            dietary_habits = st.selectbox("Dietary Habits", 
                ["Healthy", "Moderate", "Unhealthy"])
            
            suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", 
                ["Yes", "No"])
            
            study_hours = st.number_input("Study Hours per Day", 0, 24, 8)
            financial_stress = st.slider("Financial Stress Level (1-5)", 1, 5, 3)
            
            family_history = st.selectbox("Family History of Mental Illness", 
                ["Yes", "No"])
            
            submitted = st.form_submit_button("Analyze")
        
        if submitted:
            # Prepare input data
            input_data = {
                'Gender': gender,
                'Age': age,
                'Academic Pressure': academic_pressure,
                'Study Satisfaction': study_satisfaction,
                'Sleep Duration': sleep_duration,
                'Dietary Habits': dietary_habits,
                'Have you ever had suicidal thoughts ?': suicidal_thoughts,
                'Study Hours': study_hours,
                'Financial Stress': financial_stress,
                'Family History of Mental Illness': family_history
            }
            
            # Process input data
            processed_input = data_processor.preprocess_input(input_data)
            
            # Get predictions for different train-test splits
            split_ratios = [(80, 20), (70, 30), (60, 40)]
            results = {}
            
            for train_size, test_size in split_ratios:
                predictions = trainer.predict_all_models(
                    processed_input, 
                    train_size/100
                )
                results[f"{train_size}-{test_size}"] = predictions
            
            # Display results
            st.header("Analysis Results")
            
            # Plot results
            fig = plot_model_comparisons(results)
            st.plotly_chart(fig)
            
            # Display numerical results
            st.subheader("Detailed Predictions")
            for split, predictions in results.items():
                st.write(f"\nTrain-Test Split: {split}")
                for model_name, prob in predictions.items():
                    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
                    st.write(f"{model_name}: {prob:.2%} (Risk Level: {risk_level})")
            
            # Display important notice
            st.warning("""
                This screening tool is for educational purposes only and should not be used as a substitute 
                for professional medical advice, diagnosis, or treatment. If you're experiencing symptoms 
                of depression, please consult with a qualified mental health professional.
            """)
            
            # Add helpful resources
            st.info("""
                **Mental Health Resources:**
                - National Suicide Prevention Lifeline (US): 1-800-273-8255
                - Crisis Text Line: Text HOME to 741741
                - Please seek professional help if you're experiencing mental health concerns
            """)

if __name__ == "__main__":
    main()
