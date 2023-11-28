# evaluate.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from dataset import test_images, test_ages, test_genders
from config import BEST_MODEL_PATH

# Load the saved model
model = load_model(BEST_MODEL_PATH)

# Make predictions on the test set
predictions = model.predict(test_images)
predicted_ages = predictions[0]
predicted_genders = predictions[1]

# Evaluate age predictions
# If your ages were binned into categories, you'd convert the predictions back from one-hot encoding
# For simplicity here, I'm assuming it's a binary classification problem as you've set it before
predicted_age_classes = (predicted_ages > 0.5).astype(int)
actual_age_classes = test_ages

# Evaluate gender predictions
predicted_gender_classes = (predicted_genders > 0.5).astype(int)
actual_gender_classes = test_genders

# Calculate metrics
age_classification_report = classification_report(actual_age_classes, predicted_age_classes, target_names=['Young', 'Old'])
gender_classification_report = classification_report(actual_gender_classes, predicted_gender_classes, target_names=['Female', 'Male'])

age_confusion_matrix = confusion_matrix(actual_age_classes, predicted_age_classes)
gender_confusion_matrix = confusion_matrix(actual_gender_classes, predicted_gender_classes)

# Print out the metrics
print("Age Classification Report:")
print(age_classification_report)
print("Age Confusion Matrix:")
print(age_confusion_matrix)

print("\nGender Classification Report:")
print(gender_classification_report)
print("Gender Confusion Matrix:")
print(gender_confusion_matrix)

# Optionally, you can also plot confusion matrices or other metrics visually using matplotlib or seaborn
