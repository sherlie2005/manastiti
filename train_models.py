import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from utils import preprocess_training_data

# Load dataset
data = pd.read_csv('Depression Student Dataset.csv')
X, y = preprocess_training_data(data)

# Models to train
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Voting Classifier': VotingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gnb', GaussianNB())
    ], voting='soft'),
    'Stacking Classifier': StackingClassifier(estimators=[
        ('lr', LogisticRegression()),
        ('rf', RandomForestClassifier())
    ], final_estimator=GradientBoostingClassifier())
}

# Train/test splits
splits = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
models_path = 'models/'
os.makedirs(models_path, exist_ok=True)

# Train and save models
for model_name, model in models.items():
    for train_size, test_size in splits:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=42)

        model_file = f"{models_path}{model_name.replace(' ', '_')}_{int(train_size * 100)}.pkl"
        
        # Train only if not already saved
        if not os.path.exists(model_file):
            model.fit(X_train, y_train)
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model {model_name} trained and saved for {int(train_size * 100)}-{int(test_size * 100)} split.")

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} ({int(train_size * 100)}-{int(test_size * 100)} split) Accuracy: {accuracy:.2f}")
