import streamlit as st
import pandas as pd
import json
import mlflow
import mlflow.sklearn


CLASS_MAP =  {
        0: "Furniture",
        1: "Office Supplies",
        2:"Technology"
    }

# -----------------------------
# Load Model + Encoding
# -----------------------------
@st.cache_resource
def load_model():
    try:
        # Find the experiment we created in train_mlflow.py
        experiment = mlflow.get_experiment_by_name("Profitability_Prediction_Class2")
        
        if experiment is None:
            st.error("⚠️ MLflow experiment not found. Please run 'python train_mlflow.py' first.")
            return None
            
        # Get the most recent run from the experiment
        df_runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        
        if df_runs.empty:
            st.error("⚠️ No MLflow runs found. Please run 'python train_mlflow.py' first.")
            return None
            
        # Extract the run ID and build the MLflow model URI
        latest_run_id = df_runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/random_forest_model"
        
        # Load the model directly from the MLflow registry
        return mlflow.sklearn.load_model(model_uri)
        
    except Exception as e:
        st.error(f"Error loading model from MLflow: {e}")
        return None

@st.cache_resource
def load_encoding():
    with open("all_encodings.json", "r") as f:
        return json.load(f)

model = load_model()
encoding_map = load_encoding()

required_columns = [
    "Ship Mode",
    "Segment",
    "City",
    "State",
    "Country",
    "Market",
    "Region",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Profit",
    "Shipping Cost",
    "Order Priority"
]

# -----------------------------
# Encoding Function
# -----------------------------
def encode_input(df, default_value=-1):
    df = df.copy()

    for col, mapping in encoding_map.items():
        if col in df.columns:
            unknown_mask = ~df[col].isin(mapping.keys())

            if unknown_mask.any():
                st.warning(f"⚠️ Unknown values found in '{col}', using fallback ({default_value})")

            df[col] = df[col].map(mapping).fillna(default_value)

    return df

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Profitability Predictor", layout="wide")
st.title("📦 Order Category Predictor (MLflow Edition)")

if model is None:
    st.stop() # Stop the app from running further if the model isn't loaded

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio("Navigation", [
    "Single Prediction",
    "Batch Prediction",
    "Insights"
])

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
if page == "Single Prediction":
    st.header("🧪 Single Prediction")

    order_priority_options = ["Critical", "High", "Low", "Medium"]
    ship_mode_options = ["First Class", "Same Day", "Second Class", "Standard Class"]
    segment_options = ["Consumer", "Corporate", "Home Office"]

    col1, col2, col3 = st.columns(3)

    with col1:
        ship_mode = st.selectbox("Ship Mode", ship_mode_options, index=1)
        segment = st.selectbox("Segment", segment_options, index=0)

    with col2:
        order_priority = st.selectbox("Order Priority", order_priority_options, index=1)

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "Ship Mode": ship_mode,
            "Segment": segment,
            "City": ["Hobart"],
            "State": ["Tasmania"],
            "Country": ["Australia"],
            "Market": ["APAC"],
            "Region": ["Oceania"],
            "Sub-Category": ["Labels"],
            "Sales": [100.0],
            "Quantity": [1],
            "Discount": [0.0],
            "Profit": [10.0],
            "Shipping Cost": [5.0],
            "Order Priority": order_priority
        })

        # Encode
        input_df = encode_input(input_df)

        # Predict
        prediction = model.predict(input_df)[0]
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
        else:
            proba = None

        label = CLASS_MAP.get(prediction, "Unknown")

        st.subheader(f"Prediction: {label}")

        if proba is not None:
            for i, p in enumerate(proba):
                class_name = CLASS_MAP.get(i, f"Class {i}")
                st.write(f"{class_name}: {p:.2f}")

# -----------------------------
# BATCH PREDICTION
# -----------------------------
elif page == "Batch Prediction":
    st.header("📂 Batch Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(df.head())

        # Validate columns
        if st.button("Run Predictions"):
            with st.spinner("Predicting..."):

                df_encoded = encode_input(df)
                predictions = model.predict(df_encoded)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(df_encoded)[:, 1]
                    df["Confidence"] = probabilities

                df["Prediction"] = pd.Series(predictions).map(CLASS_MAP)

            st.success("Done!")
            st.subheader("Results")
            st.dataframe(df)

            # Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )

# -----------------------------
# INSIGHTS
# -----------------------------
elif page == "Insights":
    st.header("📊 Insights")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": required_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(importance_df.set_index("Feature"))

    st.write("""
    **Interpretation:**
    - Features with higher importance affect predictions more
    - Unknown categories are assigned fallback values
    """)
