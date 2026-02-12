"""
Engineer Performance 360 Analytics Dashboard
Main application file
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import logging
import argparse

# Import utilities
from utils.data_processor import data_processor
from utils.analytics_engine import analytics_engine
from utils.visualizations import chart_generator
from utils.ai_insights import ai_insight_generator
from utils.ml_analytics import ml_analytics
from config.settings import APP_TITLE, APP_ICON, COLOR_PALETTE

# Default data file path
DEFAULT_DATA_FILE = "Engineer_Performance.csv"

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure basic logging
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom CSS - Professional Dark Theme
st.markdown(f"""
<style>
    /* Global dark theme */
    .stApp {{
        background-color: #0f172a;
    }}
    
    /* Main header */
    .main-header {{
        font-size: 2.2rem;
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1.2rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Sub header */
    .sub-header {{
        font-size: 1.4rem;
        color: #60a5fa;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }}
    
    /* Metric cards - Glassmorphism */
    .metric-card {{
        background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(30,58,138,0.3) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(59,130,246,0.3);
        border: 1px solid rgba(255,255,255,0.2);
    }}
    
    /* Insight box - Dark theme */
    .insight-box {{
        background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(51,65,85,0.6) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.05);
    }}
    
    /* Stats card - Glassmorphism */
    .stats-card {{
        background: rgba(30,41,59,0.6);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        backdrop-filter: blur(5px);
    }}
    
    /* Tab styling - Modern */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: rgba(30,41,59,0.5);
        padding: 8px;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 10px 20px;
        background-color: transparent;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(59,130,246,0.1);
        color: #e2e8f0;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(59,130,246,0.4);
    }}
    
    /* Sidebar styling - Dark */
    [data-testid="stSidebar"] {{
        background-color: #1e293b;
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    
    /* Button styling */
    .stDownloadButton button, .stButton button {{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }}
    .stDownloadButton button:hover, .stButton button:hover {{
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        box-shadow: 0 4px 12px rgba(16,185,129,0.3);
        transform: translateY(-1px);
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        background-color: rgba(30,41,59,0.6);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Selectbox styling */
    .stSelectbox label, .stDateInput label, .stFileUploader label {{
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: rgba(30,41,59,0.6);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-color: #3b82f6 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Dynamic CSS based on theme
def get_theme_css(theme):
    """Generate CSS based on selected theme"""
    if theme == 'light':
        return """
<style>
    .stApp { background-color: #f8fafc; }
    .main-header { 
        font-size: 2.2rem; color: #1e293b; font-weight: 700; margin-bottom: 1.5rem; 
        text-align: center; padding: 1.2rem;
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 12px; box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        border: 1px solid rgba(255,255,255,0.3); color: white;
    }
    .sub-header { font-size: 1.4rem; color: #1e40af; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #3b82f6; }
    .metric-card { background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(147,197,253,0.2) 100%); padding: 1.5rem; border-radius: 12px; color: #1e293b; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid rgba(59,130,246,0.2); }
    .insight-box { background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(241,245,249,0.8) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid rgba(203,213,225,0.5); color: #1e293b; }
    .stats-card { background: rgba(255,255,255,0.9); padding: 1.2rem; border-radius: 10px; border: 1px solid rgba(203,213,225,0.5); box-shadow: 0 2px 8px rgba(0,0,0,0.06); color: #1e293b; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid rgba(203,213,225,0.5); }
</style>
        """
    else:
        return """
<style>
    .stApp { background-color: #0f172a; }
    .main-header { font-size: 2.2rem; color: #ffffff; font-weight: 700; margin-bottom: 1.5rem; text-align: center; padding: 1.2rem; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); }
    .sub-header { font-size: 1.4rem; color: #60a5fa; font-weight: 600; margin-top: 1.5rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #3b82f6; }
    .metric-card { background: linear-gradient(135deg, rgba(59,130,246,0.2) 0%, rgba(30,58,138,0.3) 100%); padding: 1.5rem; border-radius: 12px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.1); }
    .insight-box { background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(51,65,85,0.6) 100%); padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); color: #e2e8f0; }
    .stats-card { background: rgba(30,41,59,0.6); padding: 1.2rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.2); color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #1e293b; border-right: 1px solid rgba(255,255,255,0.05); }
</style>
        """

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'metrics_df' not in st.session_state:
    st.session_state.metrics_df = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None
if 'last_upload' not in st.session_state:
    st.session_state.last_upload = None

# Apply CSS based on current theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
current_theme = st.session_state.theme
st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

# Inform chart generator of theme so plotly charts match Streamlit theme
try:
    chart_generator.theme = current_theme
except Exception:
    pass

# Add theme toggle
with st.sidebar:
    theme_toggle = st.selectbox("🎨 Theme", ["Dark", "Light"], key="theme_toggle")
    if theme_toggle == "Light":
        st.session_state.theme = "light"
    else:
        st.session_state.theme = "dark"
    st.markdown("---")
    
    # Data Source Section
    st.markdown("### 📂 Data Source")
    
    # File uploader (override default file)
    uploaded_file = st.file_uploader("Upload CSV / Excel (optional)", type=['csv', 'xlsx', 'xls'], help="Upload your data to override the default dataset.")

    # Show current data file
    st.info(f"📁 Loading from: `{DEFAULT_DATA_FILE}`")
    
    # Auto-load data on first run
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Auto-loading data..."):
            try:
                # If user uploaded a file, load that first
                if uploaded_file is not None:
                    df_raw = data_processor.load_data_file(uploaded_file)
                    st.session_state.last_upload = datetime.now()
                    st.success("✅ Uploaded file loaded into session")
                else:
                    # Load data from default file path
                    df_raw = data_processor.load_from_path(DEFAULT_DATA_FILE)
                
                if df_raw is not None:
                    # Show raw data info
                    st.write(f"� Raw data: {len(df_raw):,} rows, {len(df_raw.columns)} columns")
                    
                    # Preprocess data
                    df = data_processor.preprocess(df_raw)
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.filtered_df = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.data_summary = data_processor.get_data_summary(df)
                    st.session_state.last_upload = datetime.now()
                    
                    st.success(f"✅ Data loaded successfully!")
                    st.info(f"📊 Records: {len(df):,}")
                    
                    # Debug info - show if data was filtered
                    if len(df) < len(df_raw):
                        st.warning(f"⚠️ {len(df_raw) - len(df):,} rows were filtered out during preprocessing")
                    
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.session_state.data_loaded = False
    else:
        st.success("✅ Data already loaded")
        if st.button("🔄 Reload Data", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
    
    # Filter section (only if data is loaded)
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        # Start with original data
        filtered_df = st.session_state.df.copy()
        
        # Date filter
        if 'Date' in filtered_df.columns:
            st.markdown("#### 📅 Date Range")
            valid_dates = filtered_df['Date'].dropna()
            if len(valid_dates) > 0:
                min_date = pd.to_datetime(valid_dates.min())
                max_date = pd.to_datetime(valid_dates.max())
                
                date_range = st.date_input(
                    "Select period",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key='date_filter'
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    date_mask = filtered_df['Date'].notna()
                    filtered_df = filtered_df[
                        date_mask &
                        (pd.to_datetime(filtered_df['Date']) >= pd.to_datetime(start_date)) &
                        (pd.to_datetime(filtered_df['Date']) <= pd.to_datetime(end_date))
                    ]
            else:
                st.warning("⚠️ No valid dates found in data")
        
        # Engineer filter
        if 'Owner' in filtered_df.columns:
            st.markdown("#### 👨‍🔧 Engineer")
            engineers = sorted(st.session_state.df['Owner'].unique().tolist())
            selected_engineers = st.multiselect(
                "Select engineer(s)",
                engineers,
                default=[],
                key='engineer_filter'
            )
            
            # Debug info
            if selected_engineers:
                st.caption(f"✓ {len(selected_engineers)} engineer(s) selected")
                filtered_df = filtered_df[filtered_df['Owner'].isin(selected_engineers)]
                st.caption(f"📊 {len(filtered_df)} records match")
        
        # Task type filter
        if 'Type' in filtered_df.columns:
            st.markdown("#### 📋 Task Type")
            task_types = ['All'] + sorted(st.session_state.df['Type'].dropna().unique().tolist())
            selected_type = st.selectbox(
                "Select task type",
                task_types,
                key='type_filter'
            )
            
            if selected_type != 'All':
                filtered_df = filtered_df[filtered_df['Type'] == selected_type]
        
        # Status filter
        if 'Status_Category' in filtered_df.columns:
            st.markdown("#### ⚡ Status")
            statuses = ['All'] + sorted(st.session_state.df['Status_Category'].unique().tolist())
            selected_status = st.selectbox(
                "Select status",
                statuses,
                key='status_filter'
            )
            
            if selected_status != 'All':
                filtered_df = filtered_df[filtered_df['Status_Category'] == selected_status]
        
        # Store filtered result
        st.session_state.filtered_df = filtered_df
        
        # Reset filters button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.filtered_df = st.session_state.df.copy()
            st.rerun()
        
        # Data info
        st.markdown("---")
        st.markdown("### 📊 Current View")
        
        current_df = st.session_state.filtered_df if st.session_state.filtered_df is not None else st.session_state.df
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tasks", f"{len(current_df):,}")
        with col2:
            if 'Owner' in current_df.columns:
                st.metric("Engineers", f"{current_df['Owner'].nunique()}")
        
        if 'Duration_Hours' in current_df.columns:
            st.metric("Total Hours", f"{current_df['Duration_Hours'].sum():.1f}")
    
    else:
        # Data loaded message
        if st.session_state.data_loaded:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <p style="color: #10B981; font-size: 1rem;">
                    ✅ Data loaded! Use the filters above to analyze.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 1rem;">
                <p style="color: #F59E0B; font-size: 1rem;">
                    ⏳ Waiting for data to load...
                </p>
            </div>
            """, unsafe_allow_html=True)

# Main content area
# Polished header: title, last upload, and quick actions
with st.container():
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
    with hcol2:
        if st.session_state.last_upload:
            st.markdown(f"**Last load:** {st.session_state.last_upload.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.markdown("**Last load:** -")
        if st.button("🔄 Refresh", key="top_refresh"):
            st.session_state.data_loaded = False
            st.experimental_rerun()

if st.session_state.data_loaded and st.session_state.filtered_df is not None:
    
    # Get current dataframe
    df = st.session_state.filtered_df.copy()
    
    # Calculate metrics with ML analytics
    with st.spinner("📊 Calculating performance metrics & ML analytics..."):
        metrics_df = analytics_engine.calculate_engineer_metrics(df)
        
        # Apply ML analytics
        metrics_df = ml_analytics.engineer_clustering(metrics_df)
        metrics_df = ml_analytics.detect_anomalies(metrics_df)
        metrics_df = ml_analytics.predict_future_performance(metrics_df)
        
        st.session_state.metrics_df = metrics_df
        
        temporal_metrics = analytics_engine.calculate_temporal_metrics(df)
        task_type_metrics = analytics_engine.calculate_task_type_metrics(df)
        performance_insights = analytics_engine.generate_performance_insights(metrics_df)
        
        # Get 360-degree insights
        insights_360 = ml_analytics.get_360_insights(df, metrics_df)
        workload_analysis = ml_analytics.calculate_workload_balance(metrics_df)
    
    # KPI Row
    st.markdown('<h2 class="sub-header">📊 Key Performance Indicators</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_tasks = len(df)
        st.markdown(f"""
        <div class="metric-card" style="background: {COLOR_PALETTE['gradient_blue']};">
            <h3 style="margin:0; font-size:0.9rem; opacity:0.9;">Total Tasks</h3>
            <h2 style="margin:0; font-size:2.2rem;">{total_tasks:,}</h2>
            <p style="margin:0; font-size:0.8rem; opacity:0.8;">analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_engineers = df['Owner'].nunique() if 'Owner' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card" style="background: {COLOR_PALETTE['gradient_green']};">
            <h3 style="margin:0; font-size:0.9rem; opacity:0.9;">Active Engineers</h3>
            <h2 style="margin:0; font-size:2.2rem;">{active_engineers}</h2>
            <p style="margin:0; font-size:0.8rem; opacity:0.8;">field staff</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        completed = len(df[df['Status_Category'].str.contains('DONE', case=False, na=False)]) if 'Status_Category' in df.columns else 0
        completion_pct = (completed / total_tasks * 100) if total_tasks > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="background: {COLOR_PALETTE['gradient_cyan']};">
            <h3 style="margin:0; font-size:0.9rem; opacity:0.9;">Completion Rate</h3>
            <h2 style="margin:0; font-size:2.2rem;">{completion_pct:.1f}%</h2>
            <p style="margin:0; font-size:0.8rem; opacity:0.8;">{completed:,} completed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_duration = df['Duration_Hours'].mean() if 'Duration_Hours' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card" style="background: {COLOR_PALETTE['gradient_orange']};">
            <h3 style="margin:0; font-size:0.9rem; opacity:0.9;">Avg Duration</h3>
            <h2 style="margin:0; font-size:2.2rem;">{avg_duration:.1f}h</h2>
            <p style="margin:0; font-size:0.8rem; opacity:0.8;">per task</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        unique_accounts = df['Account'].nunique() if 'Account' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card" style="background: {COLOR_PALETTE['primary']};">
            <h3 style="margin:0; font-size:0.9rem; opacity:0.9;">Clients Served</h3>
            <h2 style="margin:0; font-size:2.2rem;">{unique_accounts}</h2>
            <p style="margin:0; font-size:0.8rem; opacity:0.8;">active accounts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Insights Section
    st.markdown('<h2 class="sub-header">🤖 AI-Powered Insights</h2>', unsafe_allow_html=True)
    
    with st.spinner("🧠 Generating AI insights..."):
        ai_summary = ai_insight_generator.generate_performance_summary(df, metrics_df)
        predictions = ai_insight_generator.predict_performance_trend(metrics_df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="insight-box">
            <h4 style="margin-top:0; color: {COLOR_PALETTE['primary']};">🔍 Performance Analysis</h4>
            <div style="font-size: 1rem; line-height: 1.6;">
                {ai_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin-top:0; color: {COLOR_PALETTE['primary']};">📈 Performance Predictions</h4>
            <p><strong>Trend:</strong> <span style="color: {COLOR_PALETTE['success' if predictions['trend'] == 'Improving' else 'warning' if predictions['trend'] == 'Declining' else 'info']};">{predictions['trend']}</span></p>
            <p><strong>Confidence:</strong> {predictions['confidence']}%</p>
            <p><strong>Next Month:</strong> ~{predictions['next_month_tasks']} tasks</p>
            <hr>
            <p><strong>🎯 Top Potential:</strong></p>
            <ul style="list-style-type: none; padding-left: 0;">
                {''.join([f"<li>• {e['Engineer']}: {e['Performance_Score']:.1f}%</li>" for e in predictions['top_potential'][:2]])}
            </ul>
            <p><strong>⚠️ At Risk:</strong> {len(predictions['at_risk'])} engineer(s)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance insights from analytics engine
    if performance_insights:
        with st.expander("📋 Quick Performance Insights", expanded=False):
            for insight in performance_insights:
                st.markdown(insight)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👨‍🔧 Engineer Performance",
        "📋 Task Analytics", 
        "⏱️ Time Analysis",
        "🏢 Account Analysis",
        "📈 Trends & Predictions",
        "🧠 360° ML Analytics"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Engineer Performance Ranking</h2>', unsafe_allow_html=True)
        
        if not metrics_df.empty:
            # Performance ranking chart
            fig_ranking = chart_generator.create_performance_ranking_chart(metrics_df, top_n=10)
            st.plotly_chart(fig_ranking, use_container_width=True, key="perf_ranking_chart")
            
            # Performance matrix
            st.markdown("<h3 style='margin-top:2rem;'>Performance Matrix</h3>", unsafe_allow_html=True)
            fig_matrix = chart_generator.create_performance_matrix(metrics_df)
            st.plotly_chart(fig_matrix, use_container_width=True, key="perf_matrix_chart")
            
            # Engineer details table
            st.markdown("<h3 style='margin-top:2rem;'>📋 Engineer Performance Summary</h3>", unsafe_allow_html=True)
            
            display_cols = [
                'Engineer', 'Total_Tasks', 'Completed_Tasks', 'Completion_Rate',
                'Avg_Duration_Hours', 'High_Priority_Tasks', 'Accounts_Served',
                'Efficiency_Score', 'Performance_Score'
            ]
            
            display_df = metrics_df[display_cols].copy()
            
            # Color coding function
            def color_performance(val):
                if val >= 80:
                    return 'background-color: #10B981; color: white'
                elif val >= 60:
                    return 'background-color: #FBBF24; color: black'
                elif val >= 40:
                    return 'background-color: #F59E0B; color: white'
                else:
                    return 'background-color: #EF4444; color: white'
            
            styled_df = display_df.style.applymap(
                color_performance,
                subset=['Performance_Score', 'Efficiency_Score', 'Completion_Rate']
            ).format({
                'Completion_Rate': '{:.1f}%',
                'Avg_Duration_Hours': '{:.1f}',
                'Efficiency_Score': '{:.1f}%',
                'Performance_Score': '{:.1f}%'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=450)
            
            # Individual engineer feedback
            with st.expander("🤖 Personalized Engineer Feedback", expanded=False):
                selected_eng = st.selectbox(
                    "Select engineer for personalized feedback:",
                    metrics_df['Engineer'].tolist()
                )
                
                if selected_eng:
                    eng_data = metrics_df[metrics_df['Engineer'] == selected_eng].iloc[0]
                    feedback = ai_insight_generator.generate_engineer_feedback(eng_data)
                    
                    st.markdown(f"""
                    <div style="background-color: {COLOR_PALETTE['light']}; padding: 1.5rem; border-radius: 10px;">
                        {feedback.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.info("No engineer performance data available for the selected filters")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Task Type Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_task = chart_generator.create_task_type_distribution(df)
            st.plotly_chart(fig_task, use_container_width=True, key="task_type_dist_chart")
        
        with col2:
            fig_status = chart_generator.create_status_distribution(df)
            st.plotly_chart(fig_status, use_container_width=True, key="status_dist_chart")
        
        # Priority distribution
        st.markdown("<h3 style='margin-top:2rem;'>Priority Level Distribution</h3>", unsafe_allow_html=True)
        fig_priority = chart_generator.create_priority_distribution(df)
        st.plotly_chart(fig_priority, use_container_width=True, key="priority_dist_chart")
        
        # Task type metrics table
        if not task_type_metrics.empty:
            st.markdown("<h3 style='margin-top:2rem;'>📊 Task Type Performance</h3>", unsafe_allow_html=True)
            st.dataframe(
                task_type_metrics.style.format({
                    'Avg_Duration': '{:.1f}',
                    'Completion_Rate': '{:.1f}%'
                }),
                use_container_width=True
            )
    
    with tab3:
        st.markdown('<h2 class="sub-header">⏱️ Time & Productivity Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_duration = chart_generator.create_duration_histogram(df)
            st.plotly_chart(fig_duration, use_container_width=True, key="duration_hist_chart")
        
        with col2:
            fig_workload = chart_generator.create_workload_distribution(df)
            st.plotly_chart(fig_workload, use_container_width=True, key="workload_dist_chart")
        
        # Daily trend
        st.markdown("<h3 style='margin-top:2rem;'>Daily Task Volume Trend</h3>", unsafe_allow_html=True)
        fig_trend = chart_generator.create_daily_trend_chart(df)
        st.plotly_chart(fig_trend, use_container_width=True, key="daily_trend_chart")
        
        # Hourly distribution
        if 'Hour' in df.columns:
            st.markdown("<h3 style='margin-top:2rem;'>Peak Working Hours</h3>", unsafe_allow_html=True)
            
            hour_counts = df['Hour'].value_counts().sort_index().reset_index()
            hour_counts.columns = ['Hour', 'Task Count']
            
            fig_hour = px.bar(
                hour_counts,
                x='Hour',
                y='Task Count',
                title='Task Volume by Hour of Day',
                color='Task Count',
                color_continuous_scale='viridis'
            )
            fig_hour.update_layout(height=400)
            st.plotly_chart(fig_hour, use_container_width=True, key="hourly_dist_chart")
    
    with tab4:
        st.markdown('<h2 class="sub-header">🏢 Account & Client Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_accounts = chart_generator.create_account_distribution(df)
            st.plotly_chart(fig_accounts, use_container_width=True, key="account_dist_chart")
        
        with col2:
            if not metrics_df.empty:
                # Engineer-Account relationship
                eng_account = metrics_df.nlargest(15, 'Accounts_Served')[
                    ['Engineer', 'Accounts_Served', 'Total_Tasks']
                ].copy()
                
                fig_eng_account = px.bar(
                    eng_account,
                    x='Engineer',
                    y='Accounts_Served',
                    title='Engineers by Accounts Served',
                    color='Total_Tasks',
                    color_continuous_scale='greens',
                    text='Accounts_Served'
                )
                fig_eng_account.update_traces(textposition='outside')
                fig_eng_account.update_layout(height=500)
                st.plotly_chart(fig_eng_account, use_container_width=True, key="eng_account_chart")
        
        # Account performance
        if 'Account' in df.columns and not metrics_df.empty:
            st.markdown("<h3 style='margin-top:2rem;'>📊 Top Accounts by Engineer Coverage</h3>", unsafe_allow_html=True)
            
            account_eng = df.groupby('Account').agg({
                'Owner': 'nunique',
                'Duration_Hours': 'sum'
            }).reset_index()
            account_eng.columns = ['Account', 'Engineers', 'Total_Hours']
            account_eng = account_eng.nlargest(10, 'Engineers')
            
            fig_account_eng = px.bar(
                account_eng,
                x='Account',
                y='Engineers',
                title='Accounts by Number of Engineers',
                color='Total_Hours',
                color_continuous_scale='blues',
                text='Engineers'
            )
            fig_account_eng.update_traces(textposition='outside')
            fig_account_eng.update_layout(height=400)
            st.plotly_chart(fig_account_eng, use_container_width=True, key="account_eng_chart")
    
    with tab5:
        st.markdown('<h2 class="sub-header">📈 Trends & Predictive Analytics</h2>', unsafe_allow_html=True)
        
        # Performance prediction
        st.markdown("<h3 style='margin-top:1rem;'>🎯 Performance Prediction Model</h3>", unsafe_allow_html=True)
        
        if not metrics_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <h4 style="color: {COLOR_PALETTE['primary']};">📊 Current Period</h4>
                    <h2>{len(df):,}</h2>
                    <p>total tasks</p>
                    <h3>{completion_pct:.1f}%</h3>
                    <p>completion rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <h4 style="color: {COLOR_PALETTE['primary']};">🔮 Next Period</h4>
                    <h2>{predictions['next_month_tasks']}</h2>
                    <p>predicted tasks</p>
                    <h3>{min(100, completion_pct + 2):.1f}%</h3>
                    <p>target completion</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stats-card">
                    <h4 style="color: {COLOR_PALETTE['primary']};">🎯 Performance Gap</h4>
                    <h2>{len(metrics_df[metrics_df['Performance_Score'] < 60])}</h2>
                    <p>engineers below target</p>
                    <h3>{metrics_df['Performance_Score'].mean():.1f}%</h3>
                    <p>avg performance</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Improvement opportunities
            st.markdown("<h3 style='margin-top:2rem;'>💡 Improvement Opportunities</h3>", unsafe_allow_html=True)
            
            poor_performers = metrics_df[metrics_df['Performance_Score'] < 50]
            slow_engineers = metrics_df[metrics_df['Avg_Duration_Hours'] > metrics_df['Avg_Duration_Hours'].quantile(0.75)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: {COLOR_PALETTE['light']}; padding: 1rem; border-radius: 10px;">
                    <h4 style="color: {COLOR_PALETTE['danger']};">⚠️ At-Risk Engineers</h4>
                    <p>{len(poor_performers)} engineer(s) need immediate attention</p>
                    <ul>
                        {''.join([f"<li><strong>{e}</strong>: {s:.1f}% performance</li>" 
                                 for e, s in zip(poor_performers['Engineer'].head(3), 
                                                poor_performers['Performance_Score'].head(3))])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: {COLOR_PALETTE['light']}; padding: 1rem; border-radius: 10px;">
                    <h4 style="color: {COLOR_PALETTE['warning']};">⏱️ Efficiency Improvement</h4>
                    <p>{len(slow_engineers)} engineer(s) take longer than average</p>
                    <ul>
                        {''.join([f"<li><strong>{e}</strong>: {d:.1f}h avg</li>" 
                                 for e, d in zip(slow_engineers['Engineer'].head(3),
                                                slow_engineers['Avg_Duration_Hours'].head(3))])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with tab6:
        st.markdown('<h2 class="sub-header">🧠 360° ML-Powered Analytics</h2>', unsafe_allow_html=True)
        
        if not metrics_df.empty:
            # ML Performance Clusters
            st.markdown("<h3 style='margin-top:1rem;'>🎯 Performance Clusters (K-Means ML)</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Cluster_Label' in metrics_df.columns:
                    cluster_counts = metrics_df['Cluster_Label'].value_counts()
                    for cluster, count in cluster_counts.items():
                        color = '#10b981' if 'Elite' in cluster else '#3b82f6' if 'High' in cluster else '#f59e0b' if 'Average' in cluster else '#ef4444'
                        st.markdown(f"""
                        <div style="background: {color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {color}; margin-bottom: 0.5rem;">
                            <strong>{cluster}</strong><br>
                            <span style="font-size: 1.5rem; color: {color};">{count}</span> engineers
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stats-card">
                    <h4 style="color: #8b5cf6;">🧠 ML Insights</h4>
                    <p><strong>Clustering Algorithm:</strong> K-Means</p>
                    <p><strong>Anomaly Detection:</strong> Isolation Forest</p>
                    <p><strong>Prediction Model:</strong> Linear Regression</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                if workload_analysis:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h4 style="color: #8b5cf6;">⚖️ Workload Balance</h4>
                        <h2>{workload_analysis.get('balance_score', 0):.1f}%</h2>
                        <p>{workload_analysis.get('status', 'N/A')}</p>
                        <hr>
                        <p>CV: {workload_analysis.get('coefficient_of_variation', 0):.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Anomaly Detection
            if 'Is_Anomaly' in metrics_df.columns:
                st.markdown("<h3 style='margin-top:2rem;'>⚠️ Anomaly Detection Results</h3>", unsafe_allow_html=True)
                
                anomalies = metrics_df[metrics_df['Is_Anomaly'] == True]
                if not anomalies.empty:
                    st.warning(f"🔍 {len(anomalies)} performance anomalies detected")
                    
                    anomaly_display = anomalies[['Engineer', 'Performance_Score', 'Completion_Rate', 
                                                  'Avg_Duration_Hours', 'Anomaly_Type']].head(10)
                    st.dataframe(anomaly_display, use_container_width=True)
                else:
                    st.success("✅ No significant performance anomalies detected")
            
            # 360° Insights Summary
            st.markdown("<h3 style='margin-top:2rem;'>📊 Comprehensive 360° Insights</h3>", unsafe_allow_html=True)
            
            if insights_360:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="stats-card">
                        <h4 style="color: #10b981;">✅ Team Strengths</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                    """, unsafe_allow_html=True)
                    for strength in insights_360.get('strengths', []):
                        st.markdown(f"<li>{strength}</li>", unsafe_allow_html=True)
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="stats-card">
                        <h4 style="color: #f59e0b;">📈 Improvement Areas</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                    """, unsafe_allow_html=True)
                    for area in insights_360.get('improvement_areas', []):
                        st.markdown(f"<li>{area}</li>", unsafe_allow_html=True)
                    st.markdown("</ul></div>", unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("<h4 style='margin-top:1.5rem; color: #3b82f6;'>💡 AI + ML Recommendations</h4>", unsafe_allow_html=True)
                for rec in insights_360.get('recommendations', []):
                    st.markdown(f"• {rec}")
        else:
            st.info("No data available for ML analytics")
    
    # Export section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">📤 Export & Reports</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Export filtered data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"engineer_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export metrics
        if not metrics_df.empty:
            metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Performance Metrics",
                data=metrics_csv,
                file_name=f"engineer_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        # Generate report
        if st.button("📄 Generate Report", use_container_width=True):
            st.info("Report generation feature coming soon!")
    
    with col4:
        # Refresh data
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()
    
    # Data quality summary
    with st.expander("📋 Data Quality & Summary", expanded=False):
        if st.session_state.data_summary:
            summary = st.session_state.data_summary
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", summary['total_rows'])
                st.metric("Total Columns", summary['total_columns'])
            with col2:
                st.metric("Missing Values", summary['missing_values'])
                st.metric("Duplicate Rows", summary['duplicates'])
            with col3:
                st.metric("Encoding", summary['encoding'])
                if summary['date_range']:
                    min_date = summary['date_range']['min']
                    max_date = summary['date_range']['max']
                    # Handle both datetime and date objects
                    min_str = min_date.date() if hasattr(min_date, 'date') else min_date
                    max_str = max_date.date() if hasattr(max_date, 'date') else max_date
                    st.metric("Date Range", f"{min_str} to {max_str}")
        
        st.markdown("### 🔍 Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)

    # Footer
    with st.container():
        st.markdown("---")
        fcol1, fcol2 = st.columns([3, 1])
        with fcol1:
            st.caption("Engineer Performance 360 Analytics — professional dashboard.\nDeveloped by FaizanJamil.")
        with fcol2:
            st.caption(f"App version: 0.1.0")

else:
    # Welcome screen when no data is loaded
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 20px; margin: 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔧</div>
        <h2 style="color: #60a5fa; margin-bottom: 1rem; font-weight: 700;">Engineer Performance 360 Analytics</h2>
        <p style="font-size: 1.2rem; color: #94a3b8; max-width: 800px; margin: 0 auto 2rem auto;">
            Professional dashboard with AI-powered insights, predictive analytics, and comprehensive 360-degree performance analysis.
        </p>
        <p style="color: #f59e0b; font-size: 1rem;">
            ⏳ Loading data from <code>Engineer_Performance.csv</code>...
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 3rem; flex-wrap: wrap;">
            <div style="text-align: center; background: rgba(59,130,246,0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(59,130,246,0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <p style="margin: 0; color: #e2e8f0;"><strong>Performance</strong></p>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Ranking & KPIs</p>
            </div>
            <div style="text-align: center; background: rgba(16,185,129,0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(16,185,129,0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <p style="margin: 0; color: #e2e8f0;"><strong>AI Insights</strong></p>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Gemini 2.5 Flash</p>
            </div>
            <div style="text-align: center; background: rgba(245,158,11,0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(245,158,11,0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                <p style="margin: 0; color: #e2e8f0;"><strong>ML Analytics</strong></p>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Predictions & Trends</p>
            </div>
            <div style="text-align: center; background: rgba(139,92,246,0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139,92,246,0.3);">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📈</div>
                <p style="margin: 0; color: #e2e8f0;"><strong>360° View</strong></p>
                <p style="margin: 0; color: #94a3b8; font-size: 0.9rem;">Complete Analysis</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def cli_validate_data(file_path: str = DEFAULT_DATA_FILE) -> None:
    """Simple CLI helper to validate the CSV/Excel data file without launching Streamlit.

    This is useful for CI or local checks: `python app.py --check-data path/to/file.csv`
    """
    try:
        logger.info("Validating data file: %s", file_path)
        df_raw = data_processor.load_from_path(file_path)
        if df_raw is None:
            logger.error("No data loaded from %s", file_path)
            return

        df = data_processor.preprocess(df_raw)
        summary = data_processor.get_data_summary(df)
        logger.info("Loaded rows: %d, columns: %d", summary['total_rows'], summary['total_columns'])
        logger.info("Missing values: %d, Duplicates: %d", summary['missing_values'], summary['duplicates'])
    except Exception as e:
        logger.exception("Data validation failed: %s", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Engineer Performance 360 App utilities')
    parser.add_argument('--check-data', '-c', nargs='?', const=DEFAULT_DATA_FILE, help='Validate a data file and print a short summary')
    args = parser.parse_args()

    if args.check_data is not None:
        cli_validate_data(args.check_data)
    else:
        print('This file is intended to be run with Streamlit: `streamlit run app.py`')

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p style="margin: 0; font-size: 0.8rem;">
        Engineer Performance 360 Analytics Dashboard | Powered by Streamlit & Gemini AI
    </p>
    <p style="margin: 0; font-size: 0.7rem; margin-top: 0.5rem;">
        2024 | All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)