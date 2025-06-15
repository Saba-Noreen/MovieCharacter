import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load raw dataset
df = pd.read_csv('A:/saba university work/intro to data science/project/movie_characters.csv')
df.replace('?', pd.NA, inplace=True)

# --- Initial Exploration ---
print(df.head())
print(df.info())
print(df.describe())
print("Mode:\n", df.mode(numeric_only=False).iloc[0])
print("Median:\n", df.median(numeric_only=True))
print(df.isnull().sum())

# --- Visualizations ---
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Value Heatmap")
plt.show()

sns.histplot(df['credit_position'].dropna(), bins=30, kde=True)
plt.title("Credit Position Histogram")
plt.show()

sns.boxplot(x='gender', y='credit_position', data=df)
plt.title("Boxplot of Credit Position by Gender")
plt.show()

df['character_name'].value_counts().head(10).plot(kind='barh')
plt.title("Top 10 Character Names")
plt.show()

# --- Data Cleaning & Encoding ---
df['credit_position'] = pd.to_numeric(df['credit_position'], errors='coerce')
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['character_name'] = df['character_name'].fillna('unknown')
df['movie_title'] = df['movie_title'].fillna('unknown')
df['credit_position'] = df['credit_position'].fillna(df['credit_position'].mean())

# --- Label Encoding ---
gender_encoder = LabelEncoder()
char_encoder = LabelEncoder()
title_encoder = LabelEncoder()

df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])
df['character_name_encoded'] = char_encoder.fit_transform(df['character_name'])
df['movie_title_encoded'] = title_encoder.fit_transform(df['movie_title'])

# --- Save encoders ---
joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(char_encoder, "char_encoder.pkl")
joblib.dump(title_encoder, "title_encoder.pkl")

# --- Feature Scaling ---
scaler = StandardScaler()
df[['credit_position']] = scaler.fit_transform(df[['credit_position']])
joblib.dump(scaler, "credit_scaler.pkl")

# --- Model Training ---
X = df[['credit_position', 'character_name_encoded', 'movie_title_encoded']]
y = df['gender_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("📦 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Save Model ---
joblib.dump(model, "gender_classifier.pkl")

# --- Save Preprocessed Dataset ---
df.to_csv("movie_characters_cleaned_preprocessed.csv", index=False)
print("✅ All files saved.")
