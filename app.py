import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="wide")
st.title("🌸 Iris Flower Classification")
st.write("Predict the species of an Iris flower using its sepal and petal measurements.")

DATA_FILE = "Iris.csv"
MODEL_FILE = "iris_model.pkl"
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        from train_model import train
        train()
    return joblib.load(MODEL_FILE)

model = load_model()
data = pd.read_csv(DATA_FILE)

st.sidebar.header("🌱 Enter flower measurements")
sepal_length = st.sidebar.number_input("Sepal length (cm)", 0.1, 10.0, 5.1, 0.1)
sepal_width = st.sidebar.number_input("Sepal width (cm)", 0.1, 10.0, 3.5, 0.1)
petal_length = st.sidebar.number_input("Petal length (cm)", 0.1, 10.0, 1.4, 0.1)
petal_width = st.sidebar.number_input("Petal width (cm)", 0.1, 10.0, 0.2, 0.1)

if st.sidebar.button("Predict species", type="primary", use_container_width=True):
    sample = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=FEATURES)
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    st.subheader("Prediction result")
    st.success(f"🌸 Predicted species: **{prediction.replace('Iris-', '').title()}**")
    prob_df = pd.DataFrame({"Species": model.classes_, "Probability": probabilities})
    st.bar_chart(prob_df.set_index("Species"))

st.divider()
st.subheader("📊 Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Total flowers", len(data))
c2.metric("Features", len(FEATURES))
c3.metric("Species", data["Species"].nunique())
st.dataframe(data.drop(columns=["Id"], errors="ignore").head(10), use_container_width=True)

st.subheader("Species distribution")
counts = data["Species"].value_counts()
fig, ax = plt.subplots()
counts.plot(kind="bar", ax=ax)
ax.set_xlabel("Species")
ax.set_ylabel("Number of samples")
ax.set_title("Number of samples per Iris species")
plt.xticks(rotation=0)
st.pyplot(fig)
