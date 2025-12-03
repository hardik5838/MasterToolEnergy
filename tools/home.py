import streamlit as st
import os

def run():
    # --- HEADER ---
    st.title("⚡ Master Energy Management Suite")
    st.markdown("""
    Welcome to the **Asepeyo Energy Platform**. This unified toolset allows you to analyze consumption patterns, 
    audit efficiency measures, simulating loads, and forecast future demand using Artificial Intelligence.
    """)
    
    st.divider()

    # --- DASHBOARD OVERVIEW ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Analytics")
        st.info("**Tools 1 & 2**")
        st.write("""
        - **Pattern Analysis:** Visualize 3-year consumption curves.
        - **Yearly Reports:** Compare billing costs and consumption across different years and centers.
        """)

    with col2:
        st.subheader("🌱 Efficiency")
        st.info("**Tool 3**")
        st.write("""
        - **Energy Audit:** Track ROI, investment, and savings from efficiency measures.
        - **Sankey Diagrams:** Visualize investment flows.
        """)

    with col3:
        st.subheader("🚀 Advanced")
        st.info("**Tools 4 & 5**")
        st.write("""
        - **NILM Digital Twin:** Simulate load curves (HVAC, Lighting) vs Real Data.
        - **AI Forecasting:** Predict future consumption using Random Forest & Weather API.
        """)

    st.divider()

    # --- SYSTEM HEALTH CHECK ---
    st.subheader("📁 System Status")
    
    # Dynamic Path Detection (Works from root or subfolder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    data_folder = os.path.join(root_dir, 'data')

    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if files:
            st.success(f"✅ **Data Connection Active**: Found {len(files)} CSV files in `{data_folder}`")
            with st.expander("View Available Data Files"):
                st.write(files)
        else:
            st.warning(f"⚠️ **Data Folder Found but Empty**: `{data_folder}` contains no CSV files.")
    else:
        st.error(f"❌ **Connection Failed**: Data folder not found at `{data_folder}`. Please create a folder named 'data' in the root directory.")

    # --- QUICK START ---
    st.markdown("---")
    st.caption("👈 **Select a tool from the sidebar to begin.**")

if __name__ == "__main__":
    run()
