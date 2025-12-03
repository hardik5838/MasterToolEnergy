import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import requests
import glob

# --------------------------------------------------------------------------
# |                    FUNCIONES DE CARGA DE DATOS                         |
# --------------------------------------------------------------------------

@st.cache_data
def load_asepeyo_energy_data(file_path):
    """Carga y procesa el archivo de consumo energético desde una ruta."""
    try:
        df = pd.read_csv(file_path, sep=',', decimal='.')
        
        # Flexibilidad en nombres de columnas
        df.rename(columns=lambda x: x.strip(), inplace=True)
        if 'Fecha' not in df.columns:
             # Intentar encontrar columnas parecidas
             cols = [c for c in df.columns if 'fecha' in c.lower()]
             if cols: df.rename(columns={cols[0]: 'Fecha'}, inplace=True)

        if 'Energía activa (kWh)' not in df.columns:
             cols = [c for c in df.columns if 'kwh' in c.lower()]
             if cols: df.rename(columns={cols[0]: 'Energía activa (kWh)'}, inplace=True)

        if 'Fecha' not in df.columns or 'Energía activa (kWh)' not in df.columns:
            st.error(f"El archivo debe contener 'Fecha' y 'Energía activa (kWh)'.")
            return pd.DataFrame()
            
        df.rename(columns={'Fecha': 'fecha', 'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al procesar el archivo de consumo: {e}")
        return pd.DataFrame()

@st.cache_data
def load_nasa_weather_data(file_path):
    """Carga y procesa el archivo de clima histórico de NASA POWER."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        start_row = 0
        for i, line in enumerate(lines):
            # Detección flexible del encabezado
            if "YEAR" in line and "MO" in line and "DY" in line:
                start_row = i
                break
        
        df = pd.read_csv(file_path, skiprows=start_row)
        
        expected_cols = ['YEAR', 'MO', 'DY', 'HR', 'T2M']
        if not all(col in df.columns for col in expected_cols):
            # Intentar ver si falta alguna y advertir
            pass 

        df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour'}))
        df.rename(columns={'T2M': 'temperatura_c'}, inplace=True)
        df['temperatura_c'] = df['temperatura_c'].replace(-999, np.nan).ffill()
        
        return df[['fecha', 'temperatura_c']]
    except Exception as e:
        st.error(f"Error al procesar el archivo de clima de la NASA: {e}")
        return pd.DataFrame()

@st.cache_data
def get_weather_forecast(api_key, lat, lon):
    """Obtiene el pronóstico del tiempo diario desde la API de Meteosource."""
    BASE_URL = "https://www.meteosource.com/api/v1/free/point"
    params = {
        "lat": lat,
        "lon": lon,
        "sections": "daily",
        "units": "metric",
        "key": api_key
    }
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            daily_data = data.get('daily', {}).get('data', [])
            if not daily_data:
                st.error("La API no devolvió datos de pronóstico diario.")
                return pd.DataFrame()
            processed_data = []
            for day in daily_data:
                processed_data.append({
                    'fecha': day.get('day'),
                    'temp_max_c': day.get('all_day', {}).get('temperature_max'),
                    'temp_min_c': day.get('all_day', {}).get('temperature_min')
                })
            df_clima_futuro = pd.DataFrame(processed_data)
            df_clima_futuro['fecha'] = pd.to_datetime(df_clima_futuro['fecha'])
            df_clima_futuro.dropna(inplace=True)
            df_clima_futuro['temp_avg_c'] = (df_clima_futuro['temp_max_c'] + df_clima_futuro['temp_min_c']) / 2
            return df_clima_futuro
        else:
            st.error(f"Error API (Código {response.status_code}): {response.json().get('detail', 'Error desconocido')}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al conectar con la API del clima: {e}")
        return pd.DataFrame()

def crear_features_temporales(df):
    """Crea columnas de features diarias basadas en la fecha."""
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
    return df

# --------------------------------------------------------------------------
# |                          LÓGICA PRINCIPAL                              |
# --------------------------------------------------------------------------

def run():
    # --- RUTA DE DATOS CENTRALIZADA ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(os.path.dirname(current_dir)) 
    DATA_DIR = os.path.join(root_dir, 'data')

    st.sidebar.title("🔮 Configuración Predicción")
    st.sidebar.markdown("---")

    # --- CARGA DE ARCHIVOS ---
    st.sidebar.header("1. Carga de Datos Históricos")
    st.sidebar.info(f"Buscando en: {DATA_DIR}")

    selected_energy_file = None
    selected_weather_file = None

    if not os.path.exists(DATA_DIR):
        st.error(f"No se encuentra la carpeta {DATA_DIR}")
        return

    try:
        # Buscamos archivos CSV generales si no cumplen el patrón exacto, para ser más flexibles
        all_csvs = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        
        # Filtros opcionales por nombre (puedes relajarlos si los nombres cambian)
        energy_files = [os.path.basename(f) for f in all_csvs if "energy" in os.path.basename(f).lower() or "consumo" in os.path.basename(f).lower() or "lectura" in os.path.basename(f).lower()]
        weather_files = [os.path.basename(f) for f in all_csvs if "weather" in os.path.basename(f).lower() or "clima" in os.path.basename(f).lower() or "power" in os.path.basename(f).lower()]

        # Fallback: si no encuentra patrones, muestra todos los CSVs
        if not energy_files: energy_files = [os.path.basename(f) for f in all_csvs]
        if not weather_files: weather_files = [os.path.basename(f) for f in all_csvs]

        if not energy_files:
            st.sidebar.error(f"No hay archivos CSV en la carpeta data.")
        else:
            selected_energy_file = st.sidebar.selectbox("Archivo Consumo", energy_files)

        if weather_files:
            selected_weather_file = st.sidebar.selectbox("Archivo Clima Histórico", weather_files, index=0)
        else:
            st.sidebar.warning("No se encontraron archivos de clima.")
            
    except Exception as e:
        st.sidebar.error(f"Error al leer carpeta: {e}")

    st.sidebar.markdown("---")
    st.sidebar.header("2. API Meteosource")
    # Sugerencia: Puedes poner un valor por defecto vacío o uno de prueba si tienes
    api_key = st.sidebar.text_input("API Key", type="password") 
    lat = st.sidebar.text_input("Latitud", "40.4168")
    lon = st.sidebar.text_input("Longitud", "-3.7038")

    st.sidebar.markdown("---")
    st.sidebar.header("3. Ajustes")
    ocupacion_media = st.sidebar.slider("Ocupación Media (%)", 0, 100, 80)

    # --- CUERPO DE LA PÁGINA ---
    st.title("🤖 Predicción de Consumo (IA)")
    st.markdown("Algoritmo: **Random Forest Regressor**")
    st.markdown("---")

    if selected_energy_file and selected_weather_file and api_key and lat and lon:
        
        if st.button("🚀 Ejecutar Predicción"):
            with st.spinner('Entrenando modelo e invocando API...'):
                
                energy_path = os.path.join(DATA_DIR, selected_energy_file)
                weather_path = os.path.join(DATA_DIR, selected_weather_file)
                
                df_energia = load_asepeyo_energy_data(energy_path)
                df_clima_pasado = load_nasa_weather_data(weather_path)
                df_clima_futuro = get_weather_forecast(api_key, lat, lon)
                
                if df_clima_futuro.empty:
                    st.error("Fallo en la API de Clima Futuro. Revisa la Key.")
                    return
                    
                if not df_energia.empty and not df_clima_pasado.empty:
                    
                    # --- PREPROCESADO ---
                    df_historico_horario = pd.merge(df_energia, df_clima_pasado, on='fecha', how='inner')
                    
                    if df_historico_horario.empty:
                        st.error("No hay fechas coincidentes entre el archivo de energía y el de clima histórico.")
                        return

                    df_historico_horario.set_index('fecha', inplace=True)
                    
                    # Resample Diario
                    consumo_diario = df_historico_horario['consumo_kwh'].resample('D').sum()
                    clima_diario = df_historico_horario['temperatura_c'].resample('D').agg(['min', 'max', 'mean'])
                    
                    df_historico_daily = pd.concat([consumo_diario, clima_diario], axis=1)
                    df_historico_daily.rename(columns={'min': 'temp_min_c', 'max': 'temp_max_c', 'mean': 'temp_avg_c'}, inplace=True)
                    df_historico_daily.reset_index(inplace=True)
                    df_historico_daily.dropna(inplace=True)
                    
                    # Features
                    df_historico_daily = crear_features_temporales(df_historico_daily)
                    df_historico_daily['ocupacion'] = ocupacion_media
                    
                    df_futuro = df_clima_futuro.copy()
                    df_futuro = crear_features_temporales(df_futuro)
                    df_futuro['ocupacion'] = ocupacion_media
                    
                    # --- MACHINE LEARNING ---
                    features = ['temp_min_c', 'temp_max_c', 'temp_avg_c', 'dia_semana', 'dia_mes', 'mes', 'es_finde', 'ocupacion']
                    target = 'consumo_kwh'
                    
                    X = df_historico_daily[features]
                    y = df_historico_daily[target]
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    modelo.fit(X_train, y_train)

                    # --- RESULTADOS ---
                    X_futuro = df_futuro[features]
                    df_futuro['consumo_predicho_kwh'] = modelo.predict(X_futuro)
                    
                    pred_test = modelo.predict(X_test)
                    r2 = r2_score(y_test, pred_test)
                    rmse = np.sqrt(mean_squared_error(y_test, pred_test))

                    st.success("✅ Modelo Entrenado")
                    
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Precisión (R²)", f"{r2:.2f}")
                    kpi2.metric("Error (RMSE)", f"{rmse:.2f}")
                    kpi3.metric("Predicción Total (7 días)", f"{df_futuro['consumo_predicho_kwh'].sum():,.0f} kWh")
                    
                    st.divider()

                    # Gráfico
                    st.subheader("Visualización")
                    df_hist_plot = df_historico_daily[['fecha', 'consumo_kwh']].rename(columns={'consumo_kwh': 'Consumo'})
                    df_hist_plot['Tipo'] = 'Real (Histórico)'
                    
                    df_fut_plot = df_futuro[['fecha', 'consumo_predicho_kwh']].rename(columns={'consumo_predicho_kwh': 'Consumo'})
                    df_fut_plot['Tipo'] = 'Predicción'
                    
                    # Unir para plotear, mostrando solo último mes histórico para claridad
                    last_month = df_hist_plot['fecha'].max() - pd.Timedelta(days=30)
                    df_hist_short = df_hist_plot[df_hist_plot['fecha'] > last_month]
                    
                    df_plot = pd.concat([df_hist_short, df_fut_plot])
                    
                    fig = px.line(df_plot, x='fecha', y='Consumo', color='Tipo', markers=True,
                                  color_discrete_map={'Real (Histórico)': '#1f77b4', 'Predicción': '#ff7f0e'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("Datos Futuros:", df_futuro[['fecha', 'temp_avg_c', 'consumo_predicho_kwh']].round(1))

    else:
        st.info("👈 Por favor completa la configuración en la barra lateral (Archivos y API Key).")

if __name__ == "__main__":
    run()
