import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        """Preprocess the dataset by encoding categorical variables."""
        df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Sleep Duration', 'Dietary Habits', 
                             'Have you ever had suicidal thoughts ?', 
                             'Family History of Mental Illness']
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
            else:
                df[column] = self.label_encoders[column].transform(df[column])
        
        # Convert target variable to numeric
        if 'Depression' in df.columns:
            df['Depression'] = df['Depression'].map({'Yes': 1, 'No': 0})
        
        return df
    
    def preprocess_input(self, input_data):
        """Preprocess a single input instance."""
        df = pd.DataFrame([input_data])
        return self.preprocess_data(df)
    
    def get_feature_names(self):
        """Return list of feature names."""
        return ['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction',
                'Sleep Duration', 'Dietary Habits', 
                'Have you ever had suicidal thoughts ?',
                'Study Hours', 'Financial Stress',
                'Family History of Mental Illness']
