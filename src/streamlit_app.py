import os
import joblib
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "models/best_rf.joblib"
os.makedirs("models", exist_ok=True)

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # CSV’leri oku
    cal = pd.read_csv("calories.csv")
    ex = pd.read_csv("exercise.csv")
    df = ex.merge(cal, on="User_ID")
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})
    df["BMI"] = df["Weight"] / ((df["Height"]/100)**2)
    df["HRxDuration"] = df["Heart_Rate"] * df["Duration"]
    df["WeightxDuration"] = df["Weight"] * df["Duration"]
    df["Temp_centered"] = df["Body_Temp"] - df["Body_Temp"].mean()

    y = df["Calories"].values
    X = df.drop(columns=["Calories", "User_ID"])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    return pipe

model = get_model()
