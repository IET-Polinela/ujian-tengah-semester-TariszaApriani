from imblearn.over_sampling import SMOTE

# Terapkan SMOTE ke data latih
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Latih ulang model dengan data seimbang
model_smote = DecisionTreeClassifier(random_state=42)
model_smote.fit(X_train_resampled, y_train_resampled)

# Evaluasi kembali
y_pred_smote = model_smote.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_smote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_smote))
print("Accuracy:", accuracy_score(y_test, y_pred_smote))
