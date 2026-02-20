import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

st.set_page_config(page_title="Scientific Replication Engine", page_icon="🧪")

st.title("🧪 Autonomous Scientific Replication Engine")
st.write("Upload dataset + expected paper results → Detect reproducibility gaps.")

st.markdown("""
### CSV Requirements
Dataset must include target column.

Example expected results:
Regression → RMSE = 2.3  
Classification → Accuracy = 0.89
""")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

target = st.text_input("Target Column Name")
model_type = st.selectbox("Paper Model Type", ["Regression", "Classification"])
expected_score = st.number_input("Expected Result from Paper")

# ---------------- CORE ENGINE ----------------
def run_replication(df, target, model_type):
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, pred))
        metric = "RMSE"

    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        metric = "Accuracy"

    return score, metric

# ---------------- MAIN ----------------
if uploaded_file and target:

    df = pd.read_csv(uploaded_file)

    if target not in df.columns:
        st.error("Target column not found!")
    else:
        try:
            score, metric = run_replication(df, target, model_type)

            st.subheader("📊 Replication Result")
            st.write(f"Reproduced {metric}: **{round(score,4)}**")
            st.write(f"Expected {metric}: **{expected_score}**")

            gap = abs(score - expected_score)
            st.write(f"Reproducibility Gap: **{round(gap,4)}**")

            st.subheader("🔎 Diagnosis")

            if gap < 0.02:
                st.success("Paper is reproducible 👍")
            elif gap < 0.1:
                st.warning("Minor reproducibility gap ⚠️")
                st.write("Possible causes:")
                st.write("- Different preprocessing")
                st.write("- Random seed differences")
                st.write("- Train/test split mismatch")
            else:
                st.error("Major reproducibility issue 🚨")
                st.write("Possible causes:")
                st.write("- Missing preprocessing steps")
                st.write("- Data leakage in paper")
                st.write("- Undocumented hyperparameters")

        except Exception as e:
            st.error(f"Error running replication: {e}")

else:
    st.info("Upload dataset + fill fields to start.")

st.caption("Prototype for Hackathon Use 🚀")