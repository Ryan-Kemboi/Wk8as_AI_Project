import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Set path to your dataset
dataset_path = os.path.join('dataset', 'diabetes.csv')

# Load dataset
df = pd.read_csv(dataset_path)
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join('models', 'diabetes_model.pkl')
os.makedirs('models', exist_ok=True)
joblib.dump(model, model_path)

print(f"✅ Model trained and saved at '{model_path}'")
