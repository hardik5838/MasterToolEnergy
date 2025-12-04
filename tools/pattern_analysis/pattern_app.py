import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
import os
from urllib.parse import quote

# We wrap the entire logic in a function so the main app can call it
def run():
    
    # --- PATH SETUP (For Central Data) ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    # Go up two levels to find the 'data' folder
    root_dir = os.path.dirname(os.path.dirname(current_dir)) 
    data_folder = os.path.join(root_dir, 'data')

    # --- Funciones de Carga de Datos ---
   # --- REPLACE THIS FUNCTION IN YOUR CODE ---
@st.cache_data
def load_asepeyo_energy_data(file_path):
    """
    Robust loader that handles:
    - Comma vs Dot decimals ("81,00" vs "81.00")
    - Quotes around numbers
    - Different column names (Fecha y hora vs Fecha)
    """
    try:
        # 1. First attempt: Read CSV (handling bad lines safely)
        # We don't specify separator/decimal yet to handle mixed formats manually
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        
        # 2. Normalize Column Names (Strip whitespace)
        df.columns = df.columns.str.strip()
        
        # 3. Rename columns to standard internal names
        # Add any new variations you find here
        col_mapping = {
            'Fecha y hora': 'fecha',
            'Fecha': 'fecha',
            'Date': 'fecha',
            'Energía activa': 'consumo_kwh',
            'Energía activa (kWh)': 'consumo_kwh',
            'Consumo': 'consumo_kwh'
        }
        df.rename(columns=col_mapping, inplace=True)
        
        # 4. Check if required columns exist
        if 'fecha' not in df.columns or 'consumo_kwh' not in df.columns:
            st.error(f"Error: Could not find Date/Energy columns. Found: {list(df.columns)}")
            return pd.DataFrame()
            
        # 5. Clean Consumption Data (The Critical Fix)
        # If data is Object type (string) because of quotes/commas, clean it
        if df['consumo_kwh'].dtype == 'object':
            df['consumo_kwh'] = (
                df['consumo_kwh']
                .astype(str)
                .str.replace('"', '', regex=False)  # Remove quotes
                .str.replace('.', '', regex=False)  # Remove thousand separators (if any like 1.000,00)
                .str.replace(',', '.', regex=False) # Change comma to dot
            )
            # Force conversion to number
            df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')

        # 6. Clean Date Data
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        
        # 7. Final Cleanup
        df.dropna(subset=['fecha', 'consumo_kwh'], inplace=True)
        
        return df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return pd.DataFrame()

    @st.cache_data
    def load_nasa_weather_data(file_path):
        """Carga y procesa el archivo de clima histórico de NASA POWER."""
        try:
            if file_path.startswith('http'):
                response = requests.get(file_path)
                response.raise_for_status()
                lines = response.text.splitlines()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            
            start_row = 0
            for i, line in enumerate(lines):
                if "YEAR,MO,DY,HR" in line:
                    start_row = i
                    break
            
            if file_path.startswith('http'):
                from io import StringIO
                df = pd.read_csv(StringIO('\n'.join(lines[start_row:])))
            else:
                df = pd.read_csv(file_path, skiprows=start_row)
            
            expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M', 'RH2M']
            # Simple check if columns exist
            if not all(col in df.columns for col in expected_cols):
                # Allow for cases where we might have just processed it differently or header mismatch
                pass 

            df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
            df.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
            
            for col in ['temperatura_c', 'humedad_relativa']:
                df[col] = df[col].replace(-999, np.nan).ffill()
            
            return df[['fecha', 'temperatura_c', 'humedad_relativa']]
        except Exception as e:
            # st.error(f"Error al procesar el archivo de clima: {e}")
            return pd.DataFrame()

    # --- Barra Lateral Específica de esta Herramienta ---
    st.sidebar.markdown("### ⚙️ Configuración Patrones")
    
    source_type = st.sidebar.radio("Fuente de datos", ["Carpeta Central (Data)", "Subir Archivos", "GitHub Demo"])
    
    df_consumo = pd.DataFrame()
    df_clima = pd.DataFrame()

    # --- LÓGICA DE CARGA DE DATOS ---
    if source_type == "Carpeta Central (Data)":
        st.sidebar.info(f"Buscando en: {data_folder}")
        
        # Check if folder exists
        if os.path.exists(data_folder):
            files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
            if files:
                selected_file = st.sidebar.selectbox("Selecciona archivo de consumo", files)
                file_path = os.path.join(data_folder, selected_file)
                df_consumo = load_asepeyo_energy_data(file_path)
                
                # Try to find a weather file automatically or let user pick
                weather_files = [f for f in files if "POWER" in f or "clima" in f.lower()]
                if weather_files:
                    w_file = st.sidebar.selectbox("Selecciona archivo de clima", weather_files)
                    df_clima = load_nasa_weather_data(os.path.join(data_folder, w_file))
            else:
                st.sidebar.warning("No hay archivos CSV en la carpeta data.")
        else:
             st.sidebar.error("Carpeta 'data' no encontrada.")

    elif source_type == "Subir Archivos":
        uploaded_energy = st.sidebar.file_uploader("Archivo Consumo (CSV)", type="csv")
        uploaded_weather = st.sidebar.file_uploader("Archivo Clima (CSV)", type="csv")
        
        if uploaded_energy:
            df_consumo = load_asepeyo_energy_data(uploaded_energy)
        if uploaded_weather:
            import io
            content = uploaded_weather.getvalue().decode("utf-8")
            lines = content.splitlines()
            start = 0
            for i, l in enumerate(lines):
                if "YEAR,MO,DY,HR" in l:
                    start = i
                    break
            df_clima_raw = pd.read_csv(io.StringIO("\n".join(lines[start:])))
            df_clima_raw['fecha'] = pd.to_datetime(df_clima_raw[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
            df_clima_raw.rename(columns={'T2M': 'temperatura_c', 'RH2M': 'humedad_relativa'}, inplace=True)
            df_clima = df_clima_raw[['fecha', 'temperatura_c', 'humedad_relativa']]

    else:
        # Carga desde GitHub (Demo Mode)
        base_url = "https://raw.githubusercontent.com/hardik5838/EnergyPatternAnalysis/main/data/"
        file_energy = "251003 ASEPEYO - Curva de consumo ES0031405968956002BN.xlsx - Lecturas.csv"
        file_weather = "POWER_Point_Hourly_20230401_20250831_041d38N_002d18E_LST.csv"
        
        url_energy = base_url + quote(file_energy)
        url_weather = base_url + quote(file_weather)
        
        df_consumo = load_asepeyo_energy_data(url_energy)
        df_clima = load_nasa_weather_data(url_weather)

    # --- FILTROS (Solo si hay datos) ---
    if not df_consumo.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("Filtros Temporales")
        
        min_date = df_consumo['fecha'].min().date()
        max_date = df_consumo['fecha'].max().date()
        date_range = st.sidebar.date_input("Rango", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        dias = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        sel_dias = st.sidebar.multiselect("Días", list(dias.keys()), format_func=lambda x: dias[x], default=list(dias.keys()))
        sel_horas = st.sidebar.slider("Horas", 0, 23, (0, 23))
        
        st.sidebar.header("Limpieza")
        remove_base = st.sidebar.checkbox("Eliminar Base")
        val_base = float(df_consumo['consumo_kwh'].quantile(0.1)) if not df_consumo.empty else 0.0
        umbral_base = st.sidebar.number_input("Umbral Base", value=val_base) if remove_base else 0
        
        remove_peak = st.sidebar.checkbox("Eliminar Picos")
        umbral_pico = st.sidebar.number_input("Percentil Picos", value=99.0, min_value=90.0, max_value=100.0) if remove_peak else 100

    # --- UI PRINCIPAL ---
    st.title("📈 Análisis de Patrones Energéticos")

    if not df_consumo.empty:
        # Validar rango de fechas
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            # Aplicar filtros
            mask = (df_consumo['fecha'].dt.date >= date_range[0]) & (df_consumo['fecha'].dt.date <= date_range[1])
            mask &= df_consumo['fecha'].dt.dayofweek.isin(sel_dias)
            mask &= (df_consumo['fecha'].dt.hour >= sel_horas[0]) & (df_consumo['fecha'].dt.hour <= sel_horas[1])
            
            df_filtered = df_consumo[mask].copy()
            
            if remove_base:
                df_filtered = df_filtered[df_filtered['consumo_kwh'] > umbral_base]
            if remove_peak:
                limit = df_filtered['consumo_kwh'].quantile(umbral_pico/100)
                df_filtered = df_filtered[df_filtered['consumo_kwh'] < limit]

            if df_filtered.empty:
                st.warning("Los filtros seleccionados no han devuelto datos.")
            else:
                # --- Gráficos ---
                st.subheader("Evolución Temporal")
                st.plotly_chart(px.line(df_filtered, x='fecha', y='consumo_kwh'), use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                # Perfil Diario
                perfil_horario = df_filtered.groupby(df_filtered['fecha'].dt.hour)['consumo_kwh'].mean().reset_index()
                col1.plotly_chart(px.bar(perfil_horario, x='fecha', y='consumo_kwh', title="Perfil Diario (Hora vs kWh)", labels={'fecha': 'Hora'}), use_container_width=True)
                
                # Perfil Semanal
                perfil_semanal = df_filtered.groupby(df_filtered['fecha'].dt.dayofweek)['consumo_kwh'].mean().reset_index()
                if not perfil_semanal.empty:
                    perfil_semanal.columns = ['dia_num', 'consumo_kwh']
                    perfil_semanal['dia_nombre'] = perfil_semanal['dia_num'].map(dias)
                    perfil_semanal = perfil_semanal.sort_values('dia_num')
                    fig_semanal = px.bar(perfil_semanal, x='dia_nombre', y='consumo_kwh', title="Perfil Semanal Promedio",
                                            labels={'dia_nombre': 'Día', 'consumo_kwh': 'Consumo (kWh)'})
                    col2.plotly_chart(fig_semanal, use_container_width=True)

                # --- Correlación Clima ---
                if not df_clima.empty:
                    st.markdown("---")
                    st.subheader("Correlación con Clima")
                    df_merged = pd.merge(df_filtered, df_clima, on='fecha', how='inner')
                    
                    if not df_merged.empty:
                        c1, c2 = st.columns(2)
                        c1.plotly_chart(px.scatter(df_merged, x='temperatura_c', y='consumo_kwh', title="Consumo vs Temperatura", 
                                                    trendline="ols", trendline_color_override="red"), use_container_width=True)
                        c2.plotly_chart(px.scatter(df_merged, x='humedad_relativa', y='consumo_kwh', title="Consumo vs Humedad", 
                                                    trendline="ols", trendline_color_override="red"), use_container_width=True)
                    else:
                        st.info("No hay coincidencia exacta de fechas y horas entre consumo y clima.")
        else:
            st.info("Seleccione un rango de fecha completo (Inicio y Fin).")
    else:
        st.info("👈 Selecciona una fuente de datos en la barra lateral para comenzar.")

# This block allows you to run this file independently for testing if needed
if __name__ == "__main__":
    run()
