import os
import gdown
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page setup
st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide")
st.title("📊 Marketing Campaign Analysis Dashboard")

# --- Data loading (cached)
@st.cache_data(show_spinner=True)
def load_data(sample_size=None):
    file_id = '1jsXiPf9PpGX2nMHIk12760qwQDOiCN10'
    output = 'marketing_campaign_dataset.csv'
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=True)

    df = pd.read_csv(output, on_bad_lines='skip')
    df.columns = df.columns.str.strip()

    # Clean Acquisition_Cost column
    if 'Acquisition_Cost' in df.columns:
        df['Acquisition_Cost'] = df['Acquisition_Cost'].astype(str).replace({'\$': '', ',': ''}, regex=True)
        df['Acquisition_Cost'] = pd.to_numeric(df['Acquisition_Cost'], errors='coerce')

    # Ensure numeric conversions
    num_cols = df.select_dtypes(include=['object']).columns
    for col in num_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except Exception:
            pass

    # Sampling (if specified)
    if sample_size and df.shape[0] > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    return df

# --- Sidebar controls
st.sidebar.header("⚙️ Dashboard Options")
sample_toggle = st.sidebar.radio(
    "Choose dataset size:",
    ["Full Dataset", "Sample (max 50,000 rows)"]
)

# Decide sample size based on toggle
sample_size = None if sample_toggle == "Full Dataset" else 50000

# Load data
df = load_data(sample_size=sample_size)

st.sidebar.success(f"✅ Dataset Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# --- Show data preview
st.subheader("🔎 Dataset Preview")
st.dataframe(df.head())

# --- Basic statistics
st.subheader("📈 Summary Statistics")
st.write(df.describe(include='all'))

# --- Visualization
st.subheader("📊 Visualizations")

col1, col2 = st.columns(2)

with col1:
    if 'Age' in df.columns:
        st.write("### Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Age'].dropna(), bins=30, kde=True, ax=ax)
        st.pyplot(fig)

with col2:
    if 'Acquisition_Cost' in df.columns:
        st.write("### Acquisition Cost Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Acquisition_Cost'].dropna(), bins=30, kde=True, ax=ax)
        st.pyplot(fig)

# --- Correlation heatmap
if len(df.select_dtypes(include='number').columns) > 1:
    st.write("### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.success("✅ Dashboard Ready!")
