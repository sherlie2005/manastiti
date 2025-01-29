import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

def load_models():
    """Load all pre-trained models and encoders."""
    models = {}
    encoders = joblib.load('models/label_encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Load models for each split ratio
    for split in [80, 70, 60]:
        split_models = {}
        for model_name in ['logistic', 'svm', 'dt', 'rf', 'gbm', 'knn', 'nb', 'voting', 'stacking']:
            model_path = f'models/{model_name}_split_{split}.pkl'
            if os.path.exists(model_path):
                split_models[model_name] = joblib.load(model_path)
        models[f"{split}-{100-split}"] = split_models
    
    return models, encoders, feature_names

def preprocess_input(input_data, encoders, feature_names):
    """Preprocess input data using saved encoders."""
    df = pd.DataFrame([input_data])
    
    # Apply encoding
    for column, encoder in encoders.items():
        if column in df.columns:
            df[column] = encoder.transform(df[column])
    
    # Ensure columns are in the correct order
    return df[feature_names]

def main():
    st.title("Depression Screening Application")
    
    try:
        # Load models and encoders
        models, encoders, feature_names = load_models()
        
        # Create input form
        st.header("Please Answer the Following Questions")
        
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
            
            # Process input
            processed_input = preprocess_input(input_data, encoders, feature_names)
            
            # Get predictions from all models and splits
            results = {}
            for split, split_models in models.items():
                split_results = {}
                for name, model in split_models.items():
                    prob = model.predict_proba(processed_input)[0][1]
                    split_results[name] = prob
                results[split] = split_results
            
            # Display results
            st.header("Analysis Results")
            
            # Create heatmap
            splits = list(results.keys())
            model_names = list(results[splits[0]].keys())
            probabilities = [[results[split][model] for split in splits] for model in model_names]
            
            fig = go.Figure(data=go.Heatmap(
                z=probabilities,
                x=splits,
                y=model_names,
                colorscale='RdYlBu',
                colorbar=dict(title='Probability')
            ))
            
            fig.update_layout(
                title='Depression Risk Probability Across Models and Splits',
                xaxis_title='Train-Test Split',
                yaxis_title='Model'
            )
            
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
            
    except Exception as e:
        st.error("""
            Error: Models not found. Please ensure you have run the training script 
            (train_models.py) before running the app.
        """)
        st.exception(e)

if __name__ == "__main__":
    main()