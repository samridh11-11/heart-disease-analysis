import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
# Replace 'heart_disease_data.csv' with your actual dataset path
data_path = "heart_disease_data.csv"
heart_df = pd.read_csv(data_path)

# Display the first few rows of the dataset
print("Dataset Head:")
print(heart_df.head())

# Data preprocessing
# Check for missing values
print("\nMissing Values:")
print(heart_df.isnull().sum())

# Encode categorical variables
label_encoder = LabelEncoder()
heart_df['sex'] = label_encoder.fit_transform(heart_df['sex'])
heart_df['chest pain type'] = label_encoder.fit_transform(heart_df['chest pain type'])
heart_df['resting electrocardiographic results'] = label_encoder.fit_transform(heart_df['resting electrocardiographic results'])
heart_df['exercise induced angina'] = label_encoder.fit_transform(heart_df['exercise induced angina'])
heart_df['thal'] = label_encoder.fit_transform(heart_df['thal'])

# Feature selection
X = heart_df[['age', 'sex', 'chest pain type', 'resting blood pressure', 
               'serum cholestoral in mg/dl', 'fasting blood sugar > 120 mg/dl', 
               'resting electrocardiographic results', 'maximum heart rate achieved', 
               'exercise induced angina', 'oldpeak', 
               'slope of the peak exercise ST segment', 
               'number of major vessels (0-3) colored by flourosopy', 'thal']]
y = heart_df['target']  # Assuming 'target' is the column indicating heart disease presence

# Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'\nAccuracy of the LDA model: {accuracy:.2f}')
print('\nConfusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', class_report)

# Visualize confusion matrix
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
