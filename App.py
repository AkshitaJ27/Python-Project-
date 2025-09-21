# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from io import BytesIO

# --- Page config
st.set_page_config(page_title="Advanced Marketing Campaign Dashboard", page_icon="📊", layout="wide")

# --- Data loading (cached)
@st.cache_data(show_spinner=True)
def load_data():
    file_id = '1jsXiPf9PpGX2nMHIk12760qwQDOiCN10'
    output = 'marketing_campaign_dataset.csv'
    if not os.path.exists(output):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output, quiet=True)

    df = pd.read_csv(output, on_bad_lines='skip')

    # Sample data to prevent app crashes if it's too large
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    df.columns = df.columns.str.strip()
    # Safe conversions with checks
    if 'Acquisition_Cost' in df.columns:
        df['Acquisition_Cost'] = df['Acquisition_Cost'].astype(str).replace({'\$': '', ',': ''}, regex=True)
        df['Acquisition_Cost'] = pd.to_numeric(df['Acquisition_Cost'], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Create some derived columns if missing
    if 'Conversion_Rate' in df.columns and 'Impressions' in df.columns:
        # Ensure conversions column exists or create estimate
        if 'Conversions' not in df.columns:
            df['Conversions'] = ((df['Conversion_Rate'].astype(float) / 100) * df['Impressions']).round()
    # Engagement score numeric
    if 'Engagement_Score' in df.columns:
        df['Engagement_Score'] = pd.to_numeric(df['Engagement_Score'], errors='coerce')

    # Clean text columns if present
    for col in ['Campaign_Type', 'Target_Audience', 'Channel_Used', 'Location', 'Language', 'Company', 'Campaign_Name', 'Duration']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Fill numeric NAs with sensible defaults for analyses
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

df = load_data()

# --- Sidebar: Filters and controls
st.sidebar.header("Filters & Controls")
# Defensive: if columns missing, provide defaults
def safe_unique(col):
    return df[col].unique().tolist() if col in df.columns else []

campaign_type = st.sidebar.multiselect('Campaign Type', safe_unique('Campaign_Type'), default=safe_unique('Campaign_Type'))
target_audience = st.sidebar.multiselect('Target Audience', safe_unique('Target_Audience'), default=safe_unique('Target_Audience'))
duration = st.sidebar.multiselect('Duration', safe_unique('Duration'), default=safe_unique('Duration'))
channel_used = st.sidebar.multiselect('Channel Used', safe_unique('Channel_Used'), default=safe_unique('Channel_Used'))
location = st.sidebar.multiselect('Location', safe_unique('Location'), default=safe_unique('Location'))
language = st.sidebar.multiselect('Language', safe_unique('Language'), default=safe_unique('Language'))

# Additional controls
min_date = df['Date'].min() if 'Date' in df.columns else None
max_date = df['Date'].max() if 'Date' in df.columns else None
if min_date is not None and max_date is not None:
    date_range = st.sidebar.date_input("Date Range", value=(min_date.date(), max_date.date()))
else:
    date_range = None

# Filters applied
filtered_df = df.copy()
if 'Campaign_Type' in df.columns:
    filtered_df = filtered_df[filtered_df['Campaign_Type'].isin(campaign_type)]
if 'Target_Audience' in df.columns:
    filtered_df = filtered_df[filtered_df['Target_Audience'].isin(target_audience)]
if 'Duration' in df.columns:
    filtered_df = filtered_df[filtered_df['Duration'].isin(duration)]
if 'Channel_Used' in df.columns:
    filtered_df = filtered_df[filtered_df['Channel_Used'].isin(channel_used)]
if 'Location' in df.columns:
    filtered_df = filtered_df[filtered_df['Location'].isin(location)]
if 'Language' in df.columns:
    filtered_df = filtered_df[filtered_df['Language'].isin(language)]
if date_range and 'Date' in df.columns:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df['Date'] >= start) & (filtered_df['Date'] <= end)]

# Provide a download button for filtered data
def to_excel_bytes(df_in):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_in.to_excel(writer, index=False, sheet_name='filtered')
    return output.getvalue()

st.sidebar.markdown("---")
st.sidebar.download_button("Download filtered data (Excel)", data=to_excel_bytes(filtered_df), file_name="filtered_marketing_data.xlsx")

# --- Layout tabs
st.title("📊 Advanced Marketing Campaign Dashboard")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visual Analysis", "Segmentation & ML", "Insights & Recommendations"])

# --- Helper functions for KPIs
def safe_sum(col): return int(filtered_df[col].sum()) if col in filtered_df.columns else 0
def safe_mean(col): return float(filtered_df[col].mean()) if col in filtered_df.columns else 0.0

# ---------- OVERVIEW TAB ----------
with tab1:
    st.header("Dataset Overview & Quick KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Campaigns", f"{filtered_df.shape[0]:,}")
    col2.metric("Total Impressions", f"{safe_sum('Impressions'):,}")
    col3.metric("Total Clicks", f"{safe_sum('Clicks'):,}")
    # Conversions and CTR
    if 'Conversions' in filtered_df.columns:
        col4.metric("Total Conversions", f"{safe_sum('Conversions'):,}")
    else:
        col4.metric("Total Conversions", "N/A")
    if 'Acquisition_Cost' in filtered_df.columns and filtered_df['Acquisition_Cost'].sum() > 0:
        col5.metric("Total Spend", f"₹{filtered_df['Acquisition_Cost'].sum():,.0f}")
    else:
        col5.metric("Total Spend", "N/A")

    st.markdown("**Top 10 campaigns by ROI**")
    if 'ROI' in filtered_df.columns:
        top_roi = filtered_df.sort_values('ROI', ascending=False).head(10)[['Campaign_Name' if 'Campaign_Name' in df.columns else 'Campaign_Type','ROI','Acquisition_Cost','Clicks','Impressions']].reset_index(drop=True)
        st.dataframe(top_roi)
    else:
        st.info("ROI column not found in dataset.")

    st.markdown("**Preview (first 10 rows of filtered data)**")
    st.dataframe(filtered_df.head(10))

# ---------- VISUAL ANALYSIS TAB ----------
with tab2:
    st.header("Visual Explorations — Traditional & Unconventional")

    # Row: Time series aggregated by month (if date exists)
    if 'Date' in filtered_df.columns:
        monthly = filtered_df.copy()
        monthly['Month'] = monthly['Date'].dt.to_period('M').dt.to_timestamp()
        agg_month = monthly.groupby('Month').agg({
            'Impressions':'sum',
            'Clicks':'sum',
            'Conversions':'sum' if 'Conversions' in monthly.columns else ('Clicks' if 'Clicks' in monthly.columns else 'Impressions'),
            'Acquisition_Cost':'sum' if 'Acquisition_Cost' in monthly.columns else 'Impressions',
            'ROI':'mean' if 'ROI' in monthly.columns else 'Impressions'
        }).reset_index()
        st.subheader("Time Series: Month-level trends")
        fig_ts = go.Figure()
        if 'Impressions' in agg_month:
            fig_ts.add_trace(go.Scatter(x=agg_month['Month'], y=agg_month['Impressions'], mode='lines+markers', name='Impressions'))
        if 'Clicks' in agg_month:
            fig_ts.add_trace(go.Scatter(x=agg_month['Month'], y=agg_month['Clicks'], mode='lines+markers', name='Clicks'))
        if 'ROI' in agg_month:
            fig_ts.add_trace(go.Line(x=agg_month['Month'], y=agg_month['ROI'], name='Avg ROI', yaxis='y2'))
            fig_ts.update_layout(yaxis2=dict(overlaying='y', side='right', title='ROI'))
        fig_ts.update_layout(title='Monthly Campaign Metrics', xaxis_title='Month')
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No Date column available for time series plots.")

    # Conversion Rate by Campaign Type (stacked bar + boxplot)
    st.subheader("Campaign Type & Conversion / Cost Distributions")
    if 'Campaign_Type' in filtered_df.columns and 'Conversion_Rate' in filtered_df.columns:
        conv_by_type = filtered_df.groupby('Campaign_Type').agg({'Conversion_Rate':'mean','Acquisition_Cost':'mean','ROI':'mean'}).reset_index()
        fig_bar = px.bar(conv_by_type.sort_values('Conversion_Rate', ascending=False), x='Campaign_Type', y='Conversion_Rate', title='Mean Conversion Rate by Campaign Type', text='Conversion_Rate')
        st.plotly_chart(fig_bar, use_container_width=True)

        fig_box = px.box(filtered_df, x='Campaign_Type', y='Acquisition_Cost', title='Acquisition Cost distribution by Campaign Type', points="all")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Campaign_Type or Conversion_Rate not available for these charts.")

    # Treemap: Spend/ROI by Channel & Audience
    st.subheader("Treemap: Channel x Audience (Spend & ROI)")
    if 'Channel_Used' in filtered_df.columns:
        treemap_df = filtered_df.copy()
        if 'ROI' not in treemap_df.columns:
            treemap_df['ROI'] = 0
        treemap_agg = treemap_df.groupby(['Channel_Used', 'Target_Audience'] if 'Target_Audience' in treemap_df.columns else ['Channel_Used']).agg({
            'Acquisition_Cost':'sum',
            'ROI':'mean',
            'Impressions':'sum'
        }).reset_index()
        path_cols = ['Channel_Used', 'Target_Audience'] if 'Target_Audience' in treemap_agg.columns else ['Channel_Used']
        fig_treemap = px.treemap(treemap_agg, path=path_cols, values='Acquisition_Cost', color='ROI', hover_data=['Impressions'], title='Spend Treemap colored by ROI')
        st.plotly_chart(fig_treemap, use_container_width=True)
    else:
        st.info("Channel_Used missing — treemap not available.")

    # Funnel-ish visualization: impressions -> clicks -> conversions (aggregate)
    st.subheader("Funnel Snapshot (Impressions → Clicks → Conversions)")
    if all(c in filtered_df.columns for c in ['Impressions','Clicks','Conversions']):
        funnel = pd.DataFrame({
            'stage':['Impressions','Clicks','Conversions'],
            'count':[filtered_df['Impressions'].sum(), filtered_df['Clicks'].sum(), filtered_df['Conversions'].sum()]
        })
        fig_f = px.funnel(funnel, x='count', y='stage', title='Aggregate Funnel')
        st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.info("Impressions/Clicks/Conversions columns required for funnel.")

    # Unconventional visual: Sunburst of Location -> Channel -> Campaign_Type
    st.subheader("Sunburst: Location → Channel → Campaign Type")
    if all(c in filtered_df.columns for c in ['Location','Channel_Used','Campaign_Type']):
        sb = filtered_df.groupby(['Location','Channel_Used','Campaign_Type']).agg({'Impressions':'sum'}).reset_index()
        fig_sb = px.sunburst(sb, path=['Location','Channel_Used','Campaign_Type'], values='Impressions', title='Campaign footprint (by Impressions)')
        st.plotly_chart(fig_sb, use_container_width=True)
    else:
        st.info("Need Location, Channel_Used and Campaign_Type for sunburst.")

    # Distribution widget: histogram + violin for engagement/ROI
    st.subheader("Distribution: ROI & Engagement")
    dist_cols = [c for c in ['ROI','Engagement_Score','Conversion_Rate','Acquisition_Cost'] if c in filtered_df.columns]
    if dist_cols:
        sel = st.selectbox("Select metric for distribution", dist_cols)
        fig_hist = px.histogram(filtered_df, x=sel, nbins=40, title=f'Distribution of {sel}')
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_violin = px.violin(filtered_df, y=sel, box=True, points='all', title=f'Violin plot of {sel}')
        st.plotly_chart(fig_violin, use_container_width=True)

    # Correlation matrix heatmap (interactive)
    st.subheader("Correlation Heatmap (numeric variables)")
    numeric = filtered_df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig_corr.update_layout(height=600, title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute correlations.")

# ---------- SEGMENTATION & ML TAB ----------
with tab3:
    st.header("Segmentation, Clustering & Feature Importance")

    # Prepare features for clustering
    features = ['Conversion_Rate','ROI','Acquisition_Cost','Clicks','Impressions','Engagement_Score']
    features = [f for f in features if f in filtered_df.columns]
    if len(features) >= 2 and filtered_df.shape[0] >= 10:
        st.subheader("K-Means Clustering (campaign segments)")
        X = filtered_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Auto-find k using elbow heuristic (small)
        max_k = min(8, max(3, int(filtered_df.shape[0] / 10)))
        distortions = []
        for k in range(2, max_k+1):
            km = KMeans(n_clusters=k, random_state=42, n_init='auto')
            km.fit(X_scaled)
            distortions.append(km.inertia_)
        # Show elbow
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(range(2, max_k+1)), y=distortions, mode='lines+markers'))
        fig_elbow.update_layout(title='Elbow plot for K selection', xaxis_title='k', yaxis_title='Inertia')
        st.plotly_chart(fig_elbow, use_container_width=True)

        # Let user pick k (default 3)
        k_choice = st.slider("Choose number of clusters (k)", min_value=2, max_value=max_k, value=3)
        kmeans = KMeans(n_clusters=k_choice, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        filtered_df['cluster'] = clusters

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['pc1','pc2'])
        pca_df['cluster'] = clusters
        pca_df['Campaign'] = filtered_df['Campaign_Name'] if 'Campaign_Name' in filtered_df.columns else filtered_df.index.astype(str)

        fig_pca = px.scatter(pca_df, x='pc1', y='pc2', color=pca_df['cluster'].astype(str), hover_data=['Campaign'], title='PCA projection of clusters')
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("**Cluster summaries (mean values)**")
        st.dataframe(filtered_df.groupby('cluster')[features].mean().round(3))

        # Managerial label suggestions
        st.markdown("**Suggested cluster labels (automatic heuristics)**")
        cluster_summary = filtered_df.groupby('cluster').agg({ 'ROI':'mean' if 'ROI' in filtered_df.columns else ('Clicks' if 'Clicks' in filtered_df.columns else 'Impressions'),
                                                                'Conversion_Rate':'mean' if 'Conversion_Rate' in filtered_df.columns else 'Clicks',
                                                                'Acquisition_Cost':'mean' if 'Acquisition_Cost' in filtered_df.columns else 'Impressions'}).reset_index()
        labels = {}
        for _, row in cluster_summary.iterrows():
            lab = []
            if 'ROI' in row and row['ROI'] >= cluster_summary['ROI'].quantile(0.66):
                lab.append('High-ROI')
            elif 'ROI' in row and row['ROI'] <= cluster_summary['ROI'].quantile(0.33):
                lab.append('Low-ROI')
            if 'Conversion_Rate' in row and row['Conversion_Rate'] >= cluster_summary['Conversion_Rate'].quantile(0.66):
                lab.append('High-Conversion')
            labels[row['cluster']] = ", ".join(lab) if lab else "Moderate"
        st.write(labels)
    else:
        st.info("Not enough numeric features or records for clustering.")

    # Feature importance for ROI (simple RandomForestRegressor)
    st.subheader("Feature Importance (predicting ROI)")
    if 'ROI' in filtered_df.columns and len(features) >= 2 and filtered_df.shape[0] > 20:
        X = filtered_df[features].fillna(0)
        y = filtered_df['ROI'].fillna(0)
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        
        # --- THIS IS THE CORRECTED LINE ---
        fig_imp = px.bar(importances.reset_index().rename(columns={'index':'feature', 0:'importance'}), x='feature', y='importance', title='Feature importance for ROI prediction')
        
        st.plotly_chart(fig_imp, use_container_width=True)
        st.write("Top features:", importances.head(5).to_dict())
    else:
        st.info("ROI or sufficient sample size not available for feature importance model.")

    # Quick A/B test (two most frequent channels)
    st.subheader("Quick A/B (T-test) between top 2 channels — Conversion Rate")
    if 'Channel_Used' in filtered_df.columns and 'Conversion_Rate' in filtered_df.columns:
        top_channels = filtered_df['Channel_Used'].value_counts().nlargest(2).index.tolist()
        if len(top_channels) == 2:
            a = filtered_df[filtered_df['Channel_Used'] == top_channels[0]]['Conversion_Rate'].dropna()
            b = filtered_df[filtered_df['Channel_Used'] == top_channels[1]]['Conversion_Rate'].dropna()
            t_stat, p_val = ttest_ind(a, b, equal_var=False)
            st.write(f"Comparing `{top_channels[0]}` vs `{top_channels[1]}` — t-statistic: {t_stat:.3f}, p-value: {p_val:.3f}")
            if p_val < 0.05:
                st.success("Significant difference between channels (p < 0.05). Consider prioritizing the better-performing channel after checking spend & scale).")
            else:
                st.info("No statistically significant difference found in conversion rate between the top 2 channels.")
        else:
            st.info("Not enough distinct channels for A/B test.")
    else:
        st.info("Channel_Used and Conversion_Rate required for A/B test.")

# ---------- INSIGHTS & RECOMMENDATIONS TAB ----------
with tab4:
    st.header("Auto-generated Managerial Insights & Recommendations")
    insights = []

    # Insight 1: Top & bottom performers
    if 'ROI' in filtered_df.columns:
        best = filtered_df.sort_values('ROI', ascending=False).head(3)
        worst = filtered_df.sort_values('ROI', ascending=True).head(3)
        insights.append(("Top performers (by ROI)", best[['Campaign_Name' if 'Campaign_Name' in df.columns else 'Campaign_Type','ROI','Acquisition_Cost']].to_dict('records')))
        insights.append(("Lowest performers (by ROI)", worst[['Campaign_Name' if 'Campaign_Name' in df.columns else 'Campaign_Type','ROI','Acquisition_Cost']].to_dict('records')))
    else:
        insights.append(("ROI not available", "Cannot compute top/bottom performers by ROI."))

    # Insight 2: Cost efficiency
    if 'Acquisition_Cost' in filtered_df.columns and 'Conversions' in filtered_df.columns and filtered_df['Conversions'].sum() > 0:
        filtered_df['Cost_per_Conversion'] = filtered_df['Acquisition_Cost'] / filtered_df['Conversions'].replace(0, np.nan)
        cpc = filtered_df.groupby('Campaign_Type')['Cost_per_Conversion'].mean().sort_values()
        insights.append(("Cost per conversion by Campaign Type (lowest first)", cpc.head(5).round(2).to_dict()))
    else:
        insights.append(("Cost per conversion", "Insufficient data to compute cost per conversion."))

    # Insight 3: Under-utilized channels (high ROI but low spend)
    if 'Channel_Used' in filtered_df.columns and 'ROI' in filtered_df.columns and 'Acquisition_Cost' in filtered_df.columns:
        channel_perf = filtered_df.groupby('Channel_Used').agg({'ROI':'mean','Acquisition_Cost':'sum','Impressions':'sum'}).reset_index()
        high_roi_low_spend = channel_perf[(channel_perf['ROI'] >= channel_perf['ROI'].quantile(0.66)) & (channel_perf['Acquisition_Cost'] <= channel_perf['Acquisition_Cost'].quantile(0.33))]
        if not high_roi_low_spend.empty:
            insights.append(("Underutilized high-ROI channels", high_roi_low_spend[['Channel_Used','ROI','Acquisition_Cost']].to_dict('records')))
        else:
            insights.append(("Underutilized channels", "No channels found with high ROI & low spend in selected filter scope."))
    else:
        insights.append(("Underutilized channels", "Need Channel_Used/ROI/Acquisition_Cost to compute."))

    # Insight 4: Reallocation suggestion (heuristic)
    if 'ROI' in filtered_df.columns and 'Acquisition_Cost' in filtered_df.columns:
        # Identify bottom 20% by ROI and top 20% by ROI
        bottom = filtered_df[filtered_df['ROI'] <= filtered_df['ROI'].quantile(0.2)]
        top = filtered_df[filtered_df['ROI'] >= filtered_df['ROI'].quantile(0.8)]
        suggested_realloc = {
            "from": bottom[['Campaign_Name' if 'Campaign_Name' in df.columns else 'Campaign_Type','ROI','Acquisition_Cost']].head(5).to_dict('records'),
            "to": top[['Campaign_Name' if 'Campaign_Name' in df.columns else 'Campaign_Type','ROI','Acquisition_Cost']].head(5).to_dict('records'),
            "note": "Consider shifting small % of budget from 'from' campaigns to 'to' campaigns after validating scale and audience overlap."
        }
        insights.append(("Heuristic budget reallocation", suggested_realloc))
    else:
        insights.append(("Budget reallocation", "Insufficient ROI/spend data."))

    # Display insights in expanders
    for title, content in insights:
        with st.expander(title, expanded=False):
            st.write(content)

    # Actionable checklist
    st.subheader("Actionable Checklist (auto-generated)")
    st.markdown("""
    - Pause or A/B test the **lowest ROI** campaigns (check creative & landing pages).  
    - Reallocate small pilot budget (5-15%) to **underutilized high-ROI channels** and measure lift for 2-4 weeks.  
    - Validate top clusters' audiences: create tailored creatives for each segment (High-ROI / High-Conversion / Low-ROI).  
    - Run A/B tests on the two most active channels to confirm statistical advantage (we provided a t-test if data available).  
    - Build a monthly dashboard review with: ROI trend, top 5 campaigns, bottom 5 campaigns, and reallocation proposals.
    """)

    # Feedback & notes
    st.markdown("---")
    st.markdown("**Notes & caveats:** The recommendations are heuristic and based purely on the uploaded dataset. Always validate with experiments, check data quality & campaign objectives (e.g., brand vs. direct-response).")

# End of file
