# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Paths (relative)
PIPE_PATH = os.path.join("artifacts", "pm_pipeline.joblib")
DATA_PATH = os.path.join("data", "predictive_maintenance.csv")
FEATURE_NAMES_PATH = os.path.join("artifacts", "feature_names.json")
BACKGROUND_PATH = os.path.join("artifacts", "background.csv")

@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_pipeline(path=PIPE_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_feature_names(path=FEATURE_NAMES_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

df = load_data()
pipe = load_pipeline()
feature_names = load_feature_names()

st.title("🦾 Predictive Maintenance — Dashboard")

if df is None or pipe is None:
    st.warning("Model or dataset artifacts missing. Put `predictive_maintenance.csv` in /data and run `python src/train.py --data data/predictive_maintenance.csv --out artifacts`.")
    uploaded = st.file_uploader("Or upload dataset CSV to run app temporarily", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

target_col = "Target"
drop_candidates = ["UDI", "UID", "Failure Type"]
drop_cols = [c for c in drop_candidates if c in df.columns]
feature_columns = [c for c in df.columns if c not in drop_cols + [target_col]]

# KPIs
c1, c2, c3 = st.columns(3)
fail_count = int(df[target_col].sum()) if target_col in df.columns else 0
c1.metric("Total rows", f"{len(df):,}")
c2.metric("Failure count", f"{fail_count:,}")
c3.metric("Failure rate", f"{fail_count/len(df):.2%}" if len(df) else "N/A")

st.markdown("---")

# Sidebar
st.sidebar.header("Filters & Settings")
if "Type" in df.columns:
    sel_types = st.sidebar.multiselect("Filter by Type", options=sorted(df["Type"].unique()), default=list(df["Type"].unique()))
    df = df[df["Type"].isin(sel_types)]

st.sidebar.subheader("Cost model")
C_down = st.sidebar.number_input("Cost of Downtime ($)", value=5000.0)
C_maint = st.sidebar.number_input("Cost of Preventive Maintenance ($)", value=200.0)
recommended_threshold = float(C_maint / C_down) if C_down > 0 else 0.5
threshold = st.sidebar.slider("Decision threshold", min_value=0.0, max_value=1.0, value=recommended_threshold)
st.sidebar.markdown(f"**Recommended threshold** = {recommended_threshold:.3f}")

# EDA
st.subheader("Exploratory Analysis")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if numeric_cols:
    col = st.selectbox("Select numeric feature for histogram", numeric_cols)
    fig = px.histogram(df, x=col, nbins=50, color=target_col if target_col in df.columns else None, marginal="box")
    st.plotly_chart(fig, use_container_width=True)

# Manual sample section
st.subheader("Prediction — Manual sample")
with st.expander("Create manual sample"):
    manual_input = {}
    for col in feature_columns:
        if np.issubdtype(df[col].dtype, np.number):
            vmin, vmax = float(df[col].min()), float(df[col].max())
            vmed = float(df[col].median())
            manual_input[col] = st.number_input(col, value=vmed, min_value=vmin, max_value=vmax)
        else:
            manual_input[col] = st.selectbox(col, sorted(df[col].unique()))
    X_manual = pd.DataFrame([manual_input])

# File upload for custom data
st.subheader("Prediction — Batch")
uploaded_file = st.file_uploader("Upload a CSV to score (optional)", type=["csv"])
if uploaded_file is not None:
    df_custom = pd.read_csv(uploaded_file)
    st.write("Preview custom data:", df_custom.head())
    # run batch prediction
    if pipe is not None:
        X_all = df_custom[feature_columns]
        probs = pipe.predict_proba(X_all)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X_all)
        df_custom["predicted_failure_prob"] = probs
        flagged = df_custom[df_custom["predicted_failure_prob"] >= threshold]
        st.write(f"Flagged {len(flagged)} rows for maintenance (threshold {threshold:.3f})")
        st.dataframe(flagged.head(50))
        csv = flagged.to_csv(index=False).encode("utf-8")
        st.download_button("Download flagged CSV", csv, "flagged.csv", "text/csv")

# Single prediction
if st.button("Predict manual sample"):
    try:
        proba = float(pipe.predict_proba(X_manual)[:, 1][0])
        st.metric("Predicted failure probability", f"{proba:.2%}")
        decision = "✅ Schedule maintenance" if proba >= threshold else "🟢 No action"
        st.write("**Recommendation:**", decision)
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Explainability
st.subheader("Explainability")
# Prefer SHAP if available; fallback to feature_importances_
try:
    import shap
    if os.path.exists(BACKGROUND_PATH):
        background = pd.read_csv(BACKGROUND_PATH)
    else:
        background = df.sample(min(200, len(df)), random_state=42)[feature_columns]

    # Try SHAP using a callable that returns class-1 probability
    explainer = shap.Explainer(lambda x: pipe.predict_proba(x)[:, 1], background)
    shap_vals = explainer(X_manual)
    st.text("SHAP local explanation (bar):")
    shap.plots.bar(shap_vals, max_display=20)
    st.pyplot(bbox_inches="tight")
except Exception:
    # Fallback to model feature_importances_
    try:
        import plotly.express as px
        import json
        if feature_names is None:
            st.warning("Feature names not available. Run training script to generate artifacts/feature_names.json")
        else:
            import numpy as np
            clf = pipe.named_steps["clf"]
            fi = clf.feature_importances_
            feat_df = pd.DataFrame({"feature": feature_names, "importance": fi})
            feat_df = feat_df.sort_values("importance", ascending=False).head(30)
            fig = px.bar(feat_df, x="importance", y="feature", orientation="h", title="Top feature importances")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Explainability unavailable: {e}")
