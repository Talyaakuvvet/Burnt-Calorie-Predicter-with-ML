
import os
import sys
import joblib
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


MODEL_PATH = "models/best_rf.joblib"
os.makedirs("models", exist_ok=True)


cache_decorator = getattr(st, "cache_resource", None)
if cache_decorator is None:
    cache_decorator = lambda *a, **k: st.cache(allow_output_mutation=True)

st.set_page_config(page_title="Calories Burnt Estimator", layout="centered")
st.title("Calories Burnt Estimator")
st.caption(f"Streamlit: {st.__version__} | Python: {sys.version.split()[0]} | CWD: {os.getcwd()}")


@cache_decorator()
def get_model():
  
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # 2) CSV’ler repoda var mı?
    have_repo_csvs = os.path.exists("calories.csv") and os.path.exists("exercise.csv")

    if not have_repo_csvs:
        # 3) Yoksa kullanıcıdan iste
        st.warning("CSV cant be found. Please download calories.csv and exercise.csv.")
        cal_file = st.file_uploader("Upload calories.csv", type=["csv"], key="cal")
        ex_file  = st.file_uploader("Upload exercise.csv",  type=["csv"], key="ex")
        if cal_file is None or ex_file is None:
            st.stop()
        cal = pd.read_csv(cal_file)
        ex  = pd.read_csv(ex_file)
    else:
        cal = pd.read_csv("calories.csv")
        ex  = pd.read_csv("exercise.csv")

    # 4) Birleştir + FE
    df = ex.merge(cal, on="User_ID")
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})
    df["BMI"] = df["Weight"] / ((df["Height"]/100)**2)
    df["HRxDuration"] = df["Heart_Rate"] * df["Duration"]
    df["WeightxDuration"] = df["Weight"] * df["Duration"]
    df["Temp_centered"] = df["Body_Temp"] - df["Body_Temp"].mean()

    y = df["Calories"].values
    X = df.drop(columns=["Calories","User_ID"])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH, compress=3)  # sıkıştırarak kaydet
    return pipe

# --- MODELİ AL ---
try:
    model = get_model()
    st.success("Model is ready ✅")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# --- GİRDİLER ---
gender = st.selectbox("Gender", ["male","female"])
age = st.number_input("Age", 10, 100, 25)
height = st.number_input("Height (cm)", 120, 220, 170)
weight = st.number_input("Weight (kg)", 30, 200, 70)
duration = st.number_input("Duration (min)", 1, 240, 30)
hr = st.number_input("Heart Rate (bpm)", 40, 220, 120)
temp = st.number_input("Body Temp (°C)", 34.0, 41.0, 40.0)

# Eğitimde uygulanan FE ile aynı
gender_num = 0 if gender == "male" else 1
bmi = weight / ((height/100)**2)
hrx = hr * duration
wtx = weight * duration
temp_c = temp - 40.0  # eğitimde ortalamaya yakın sabit

row = pd.DataFrame([{
    "Gender": gender_num, "Age": age, "Height": height, "Weight": weight,
    "Duration": duration, "Heart_Rate": hr, "Body_Temp": temp,
    "BMI": bmi, "HRxDuration": hrx, "WeightxDuration": wtx, "Temp_centered": temp_c
}])

if st.button("Predict"):
    pred = float(model.predict(row)[0])
    st.metric("Estimated Calories (kcal)", f"{pred:.1f}")
