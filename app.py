import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Set wide layout
st.set_page_config(layout="wide")

# --- Load Data and Model ---
df = pd.read_csv("movie_characters_cleaned_preprocessed.csv")
model = joblib.load("gender_classifier.pkl")
scaler = joblib.load("credit_scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
char_encoder = joblib.load("char_encoder.pkl")
title_encoder = joblib.load("title_encoder.pkl")

# --- Sidebar: Personal Branding ---
st.sidebar.image("pexels-venkatesan-p-283686651-16699711.jpg", use_container_width=True)
st.sidebar.title("👩‍💻 Saba Noreen")
st.sidebar.markdown("📧 **Email:** snf191216@gmail.com")
st.sidebar.markdown("🕋️ **Signature:** *Saba N.*")

# --- Sidebar: Navigation ---
st.sidebar.markdown("---")
st.sidebar.header("📂 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA", "🤖 Model", "🎯 Predict", "📌 Conclusion"])

# --- Page: Home ---
if page == "🏠 Home":
    st.title("🎮 Movie Characters Gender Prediction App")

    st.header("📌 Introduction")
    st.markdown("""
    Welcome to the Movie Characters Analysis & Prediction App.  
    This project aims to:
    - Explore and visualize movie character data.
    - Predict character gender based on **credit position**, **character name**, and **movie title**.
    - Demonstrate a working ML pipeline in real-time.
    """)

# --- Page: EDA ---
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    with st.expander("🔍 Show Raw Data"):
        st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='gender', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Credit Position Histogram")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['credit_position'], bins=30, kde=True, ax=ax2)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Boxplot by Gender")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='gender', y='credit_position', data=df, ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.subheader("Correlation Heatmap")
        fig4, ax4 = plt.subplots()
        sns.heatmap(df[["credit_position", "gender_encoded"]].corr(), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

    st.subheader("Top 10 Most Common Character Names")
    fig5, ax5 = plt.subplots()
    df['character_name'].value_counts().head(10).plot(kind='barh', ax=ax5)
    st.pyplot(fig5)

# --- Page: Model ---
elif page == "🤖 Model":
    st.title("🤠 Gender Prediction Model")

    try:
        df['char_encoded'] = char_encoder.transform(df['character_name'])
    except:
        df['char_encoded'] = 0

    try:
        df['title_encoded'] = title_encoder.transform(df['movie_title'])
    except:
        df['title_encoded'] = 0

    credit_scaled = scaler.transform(df[['credit_position']])
    X = np.hstack((credit_scaled, df[['char_encoded', 'title_encoded']].values))
    y = df['gender_encoded']
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    st.success(f"✅ Model Accuracy: {acc:.2f}")

    with st.expander("📄 Classification Report"):
        report = classification_report(y, y_pred, output_dict=True)
        st.json(report)

# --- Page: Predict ---
elif page == "🎯 Predict":
    st.title("🎯 Predict Gender")

    st.markdown("""
    Fill in the details below to predict the **gender** of a character.
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        credit_input_text = st.number_input("Credit Position (type or use slider below)", min_value=1, max_value=100, value=50, step=1)
    with col2:
        credit_input = st.slider("Or use this slider", 1, 100, credit_input_text, step=1)

    if credit_input != credit_input_text:
        credit_input_text = credit_input

    character_name = st.text_input("Character Name", "BIANCA")
    movie_title = st.text_input("Movie Title", "10 things i hate about you")

    if st.button("Predict Gender"):
        try:
            char_encoded = char_encoder.transform([character_name])[0]
        except:
            st.warning("Character name not found in training data. Using fallback.")
            char_encoded = 0

        try:
            title_encoded = title_encoder.transform([movie_title])[0]
        except:
            st.warning("Movie title not found in training data. Using fallback.")
            title_encoded = 0

        credit_scaled = scaler.transform(np.array([[credit_input]]))[0][0]
        input_features = np.array([[credit_scaled, char_encoded, title_encoded]])

        pred = model.predict(input_features)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_features)[0]
            confidence = max(prob) * 100
            confidence_bar = {gender_encoder.inverse_transform([i])[0]: round(p * 100, 2) for i, p in enumerate(prob)}
        else:
            confidence = None
            confidence_bar = {}

        gender = gender_encoder.inverse_transform([pred])[0].lower().strip()

        if gender in ['male', 'm', '1']:
            st.success(f"👨 Predicted Gender: **Male**")
        elif gender in ['female', 'f', '0']:
            st.success(f"👩 Predicted Gender: **Female**")
        else:
            st.warning(f"Predicted Gender: **{gender.title()}**")

        if confidence is not None:
            st.info(f"Model confidence: {confidence:.2f}%")
            st.progress(int(confidence))
            st.subheader("Confidence for Each Gender")

            # Custom color bar chart using Matplotlib
            fig, ax = plt.subplots()
            labels = list(confidence_bar.keys())
            values = list(confidence_bar.values())
            colors = ['lightblue' if g.lower().startswith('m') else 'lightpink' for g in labels]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Confidence (%)")
            ax.set_ylim(0, 100)
            st.pyplot(fig)


# --- Page: Conclusion ---
elif page == "📌 Conclusion":
    st.title("📌 Conclusion")
    st.markdown("""
    - Gender can be predicted from credit position, character name, and movie title.
    - Male characters usually appear higher in the credit list.
    - This interactive app demonstrates a complete ML pipeline in Streamlit.

    **Created by:** Saba Noreen  
    **Email:** snf191216@gmail.com  
    **Signature:** *Saba N.*  
    """)
