import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(df):
    """Preprocess the dataset and save encoders."""
    df = df.copy()
    encoders = {}
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Sleep Duration', 'Dietary Habits', 
                         'Have you ever had suicidal thoughts ?', 
                         'Family History of Mental Illness']
    
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df[column] = encoders[column].fit_transform(df[column])
    
    # Convert target variable
    df['Depression'] = df['Depression'].map({'Yes': 1, 'No': 0})
    
    # Save encoders
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(encoders, 'models/label_encoders.pkl')
    
    return df

def create_models():
    """Create all models."""
    base_models = {
        'logistic': LogisticRegression(random_state=42),
        'svm': SVC(probability=True, random_state=42),
        'dt': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(random_state=42),
        'gbm': GradientBoostingClassifier(random_state=42),
        'knn': KNeighborsClassifier(),
        'nb': GaussianNB()
    }
    
    # Create ensemble models
    estimators = [(name, model) for name, model in base_models.items()]
    
    base_models['voting'] = VotingClassifier(estimators=estimators, voting='soft')
    base_models['stacking'] = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )
    
    return base_models

def train_and_save_models(data_path):
    """Train models with different splits and save them."""
    # Load and preprocess data
    df = pd.read_csv(data_path)
    processed_df = preprocess_data(df)
    
    # Prepare features and target
    X = processed_df.drop('Depression', axis=1)
    y = processed_df['Depression']
    
    # Create directory for models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Train with different splits
    split_ratios = [0.8, 0.7, 0.6]
    results = {}
    
    for ratio in split_ratios:
        print(f"\nTraining with {int(ratio*100)}-{int((1-ratio)*100)} split:")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=ratio, random_state=42
        )
        
        # Train models
        models = create_models()
        split_results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Calculate scores
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            split_results[name] = {
                'train_score': train_score,
                'test_score': test_score
            }
            
            # Save model
            joblib.dump(model, f'models/{name}_split_{int(ratio*100)}.pkl')
        
        results[f"{int(ratio*100)}-{int((1-ratio)*100)}"] = split_results
    
    # Save feature names
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    
    # Print results
    print("\nTraining Results:")
    for split, split_results in results.items():
        print(f"\nSplit {split}:")
        for model_name, scores in split_results.items():
            print(f"{model_name}:")
            print(f"  Train Score: {scores['train_score']:.4f}")
            print(f"  Test Score: {scores['test_score']:.4f}")

if __name__ == "__main__":
    train_and_save_models('depression_dataset.csv')  # Replace with your dataset path