import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os

# Set page configuration
st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the dataset from Google Drive
@st.cache_data
def load_data():
    file_id = '1jsXiPf9PpGX2nMHIk12760qwQDOiCN10'
    output = 'marketing_campaign_dataset.csv'
    # Download the file if it doesn't exist
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=False)

    df = pd.read_csv(output, on_bad_lines='skip')
    # Clean up column names
    df.columns = df.columns.str.strip()
    # Data cleaning and preprocessing
    df['Acquisition_Cost'] = df['Acquisition_Cost'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar for filters
st.sidebar.header('Filter Campaigns')
campaign_type = st.sidebar.multiselect('Campaign Type', df['Campaign_Type'].unique(), df['Campaign_Type'].unique())
target_audience = st.sidebar.multiselect('Target Audience', df['Target_Audience'].unique(), df['Target_Audience'].unique())
duration = st.sidebar.multiselect('Duration', df['Duration'].unique(), df['Duration'].unique())
channel_used = st.sidebar.multiselect('Channel Used', df['Channel_Used'].unique(), df['Channel_Used'].unique())
location = st.sidebar.multiselect('Location', df['Location'].unique(), df['Location'].unique())
language = st.sidebar.multiselect('Language', df['Language'].unique(), df['Language'].unique())

# Filter the dataframe
filtered_df = df[
    (df['Campaign_Type'].isin(campaign_type)) &
    (df['Target_Audience'].isin(target_audience)) &
    (df['Duration'].isin(duration)) &
    (df['Channel_Used'].isin(channel_used)) &
    (df['Location'].isin(location)) &
    (df['Language'].isin(language))
]

# Create tabs for different sections
st.title("📊 Marketing Campaign Analysis Dashboard")
tab1, tab2, tab3 = st.tabs(["Homepage", "Exploratory Data Analysis", "Statistical Analysis"])

# Homepage
with tab1:
    st.header("Welcome to the Marketing Campaign Dashboard!")
    st.write("""
        This dashboard provides a comprehensive analysis of the marketing campaign dataset.
        Use the filters on the left to customize the data you see in the charts.
    """)
    st.subheader("Dataset Overview")
    st.dataframe(filtered_df.head())

# Exploratory Data Analysis
with tab2:
    st.header("Exploratory Data Analysis")

    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Campaigns", f"{filtered_df.shape[0]:,}")
    with col2:
        st.metric("Total Impressions", f"{filtered_df['Impressions'].sum():,}")
    with col3:
        st.metric("Total Clicks", f"{filtered_df['Clicks'].sum():,}")
    with col4:
        st.metric("Average ROI", f"{filtered_df['ROI'].mean():.2f}")

    # Visualizations
    st.subheader("Visualizations")

    # Conversion Rate by Campaign Type
    fig1 = px.bar(
        filtered_df.groupby('Campaign_Type')['Conversion_Rate'].mean().reset_index(),
        x='Campaign_Type',
        y='Conversion_Rate',
        title='Average Conversion Rate by Campaign Type',
        color='Campaign_Type'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ROI by Channel Used
    fig2 = px.pie(
        filtered_df,
        names='Channel_Used',
        values='ROI',
        title='ROI by Channel Used',
        hole=0.3
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Clicks vs. Acquisition Cost
    fig3 = px.scatter(
        filtered_df,
        x='Clicks',
        y='Acquisition_Cost',
        color='Campaign_Type',
        title='Clicks vs. Acquisition Cost',
        hover_data=['Company']
    )
    st.plotly_chart(fig3, use_container_width=True)

# Statistical Analysis
with tab3:
    st.header("Statistical Analysis")

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(filtered_df.describe())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = filtered_df[['Conversion_Rate', 'Acquisition_Cost', 'ROI', 'Clicks', 'Impressions', 'Engagement_Score']].corr()
    fig4 = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        colorscale='Viridis'
    ))
    fig4.update_layout(title='Correlation Heatmap of Numerical Variables')
    st.plotly_chart(fig4, use_container_width=True)
