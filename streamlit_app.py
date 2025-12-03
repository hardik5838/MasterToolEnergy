import streamlit as st

# 1. PAGE CONFIGURATION
# This must be the very first command in the entire app
st.set_page_config(
    page_title="Master Energy Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS STYLING (Optional but recommended for a pro look)
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #0068c9;}
    .sub-header {font-size: 1.5rem; color: #555;}
    /* Highlight the sidebar slightly */
    [data-testid="stSidebar"] {
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

# 3. DEFINE NAVIGATION
# We map the specific Python files to the menu structure
pages = {
    "General": [
        st.Page("tools/home.py", title="Home Dashboard", icon="🏠"),
    ],
    "Analytics": [
        st.Page("tools/pattern_analysis/pattern_app.py", title="1. Pattern Analysis", icon="📈"),
        st.Page("tools/yearly_consumption/yearly_app.py", title="2. Yearly Reports", icon="📊"),
    ],
    "Efficiency & Audits": [
        st.Page("tools/energy_efficiency/efficiency_app.py", title="3. Energy Audit", icon="🌱"),
    ],
    "Advanced Engineering": [
        st.Page("tools/nilm_analysis/nilm_app.py", title="4. NILM Digital Twin", icon="🔌"),
        st.Page("tools/forecasting/forest_app.py", title="5. AI Forecasting", icon="🔮"),
    ],
}

# 4. INITIALIZE NAVIGATION
pg = st.navigation(pages)

# 5. SHARED SIDEBAR ELEMENTS
# Items here appear on ALL pages (like a Logo)
with st.sidebar:
    # You can replace this URL with your local image: st.image("assets/logo.png")
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=50) 
    st.markdown("### Energy Suite v1.0")
    st.info(f"📍 **Current Tool:**\n{pg.title}")
    st.divider()
    st.caption("Developed for Asepeyo")

# 6. RUN THE SELECTED PAGE
try:
    pg.run()
except Exception as e:
    st.error(f"Error loading the selected tool: {e}")
    st.info("Please check that the file path in app.py matches your folder structure.")
