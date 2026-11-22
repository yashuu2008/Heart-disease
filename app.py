import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Heart Disease Explorer", layout="wide")
st.title("Heart Disease Dataset Explorer")


@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")


df = load_data()

# Exploring the dataset
st.header("Exploring the Dataset")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Total Missing Values", int(df.isnull().sum().sum()))

with st.expander("Column Names & Data Types"):
    st.write("**Columns:**", list(df.columns))
    dtypes_df = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null_count": df.count(),
            "has_null": df.isnull().any(),
        }
    )
    st.dataframe(dtypes_df, use_container_width=True)

n_rows = st.slider("Number of rows to preview", min_value=1, max_value=20, value=5)
st.subheader(f"First {n_rows} Rows (`head`)")
st.dataframe(df.head(n_rows), use_container_width=True)

st.subheader(f"Last {n_rows} Rows (`tail`)")
st.dataframe(df.tail(n_rows), use_container_width=True)

st.subheader("Summary Statistics (`describe().T`)")
st.dataframe(df.describe().T, use_container_width=True)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.header("Heart Disease Prediction")

# Prepare data and train a quick baseline model
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

st.caption(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2%}")

# User input controls
col_a, col_b, col_c = st.columns(3)

with col_a:
    age = st.number_input("Age", min_value=1, max_value=120, value=54)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", options=[1, 0], index=0)
    cp = st.selectbox("Chest Pain Type (cp: 0-3)", options=[0, 1, 2, 3], index=0)
    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130
    )
    chol = st.number_input(
        "Serum Cholesterol (chol)", min_value=100, max_value=600, value=240
    )

with col_b:
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], index=0
    )
    restecg = st.selectbox(
        "Resting ECG (restecg: 0-2)", options=[0, 1, 2], index=0
    )
    thalach = st.number_input(
        "Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150
    )
    exang = st.selectbox(
        "Exercise Induced Angina (exang)", options=[0, 1], index=0
    )

with col_c:
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )
    slope = st.selectbox(
        "Slope of Peak Exercise ST (slope: 0-2)", options=[0, 1, 2], index=1
    )
    ca = st.number_input(
        "Major Vessels Colored (ca: 0-4)", min_value=0, max_value=4, value=0
    )
    thal = st.selectbox(
        "Thalassemia (thal: 0-3)", options=[0, 1, 2, 3], index=2
    )

if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ]
    )[X.columns]

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.warning(
            f"Prediction: **1 (Higher risk / disease present)** — Probability: {prob:.1%}"
        )
    else:
        st.success(
            f"Prediction: **0 (Lower risk / no disease)** — Probability: {prob:.1%}"
        )
