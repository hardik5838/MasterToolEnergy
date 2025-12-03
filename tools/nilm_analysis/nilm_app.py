import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

# ==========================================
# 1. CORE LOGIC (Simulation Engine)
# ==========================================

def generate_load_curve(hours, start, end, max_kw, ramp_up, ramp_down, dips=None):
    """
    Universal function to generate a load curve with:
    - Ramps (Up/Down)
    - Schedule (Start/End)
    - Dips (Lunch/Breaks)
    """
    if dips is None: dips = []
    curve = np.zeros(len(hours))
    
    for i, h in enumerate(hours):
        val = 0.0
        
        # 1. Active Window
        if start <= h < end:
            val = 1.0
            
            # 2. Ramp Up
            if h < (start + ramp_up):
                if ramp_up > 0: val = (h - start) / ramp_up
            
            # 3. Ramp Down
            if h >= (end - ramp_down):
                if ramp_down > 0: val = (end - h) / ramp_down
            
            # 4. Dips
            for dip in dips:
                if int(h) == int(dip['hour']):
                    val *= dip['factor']
                    
        curve[i] = np.clip(val, 0.0, 1.0) * max_kw
        
    return curve

def run_simulation(df_avg, config):
    df = df_avg.copy()
    hours = df['hora'].values
    
    # 1. BASE
    df['sim_base'] = np.full(len(hours), config['base_kw'])
    
    # 2. VENTILATION
    df['sim_vent'] = generate_load_curve(hours, config['vent_s'], config['vent_e'], config['vent_kw'], config['vent_ru'], config['vent_rd'])

    # 3. LIGHTING
    light_curve = generate_load_curve(hours, config['light_s'], config['light_e'], config['light_kw'], config['light_ru'], config['light_rd'])
    # Apply usage factor
    df['sim_light'] = light_curve * config['light_fac']
    # Security Light
    is_off = (df['sim_light'] < (config['light_kw'] * 0.1))
    df.loc[is_off, 'sim_light'] = config['light_kw'] * config['light_sec']

    # 4. THERMAL (HVAC)
    if config['hvac_mode'] == "Constant":
        df['sim_therm'] = generate_load_curve(hours, config['therm_s'], config['therm_e'], config['therm_kw'], 1, 1)
    else:
        # Weather Logic
        if 'temperatura_c' in df.columns:
            delta = (np.maximum(0, df['temperatura_c'] - config['set_c']) + 
                     np.maximum(0, config['set_h'] - df['temperatura_c']))
            raw = delta * config['therm_sens']
            sched = generate_load_curve(hours, config['therm_s'], config['therm_e'], 1.0, 1, 1)
            df['sim_therm'] = np.minimum(raw, config['therm_kw']) * sched
        else:
            df['sim_therm'] = 0

    # 5. CUSTOM PROCESSES
    total_custom = np.zeros(len(hours))
    for p in config['processes']:
        p_load = generate_load_curve(hours, p['s'], p['e'], p['kw'], p['ru'], p['rd'], p['dips'])
        col_name = f"proc_{p['name']}"
        df[col_name] = p_load
        total_custom += p_load
    df['sim_proc'] = total_custom

    # TOTAL
    df['sim_total'] = df['sim_base'] + df['sim_vent'] + df['sim_light'] + df['sim_therm'] + df['sim_proc']
    
    if 'consumo_kwh' in df.columns:
        df['diff'] = df['sim_total'] - df['consumo_kwh']
        
    return df

# ==========================================
# 2. DATA LOADING (Adapted from Tool #1)
# ==========================================

@st.cache_data
def load_hourly_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        df.rename(columns=lambda x: x.strip(), inplace=True)
        
        # Standardize columns
        if 'Fecha' in df.columns: df.rename(columns={'Fecha': 'fecha'}, inplace=True)
        if 'Energía activa (kWh)' in df.columns: df.rename(columns={'Energía activa (kWh)': 'consumo_kwh'}, inplace=True)
        # Fallback for other formats
        if 'consumo_kwh' not in df.columns:
             # Try finding a likely column
             cols = [c for c in df.columns if 'kwh' in c.lower() or 'activa' in c.lower()]
             if cols: df.rename(columns={cols[0]: 'consumo_kwh'}, inplace=True)

        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df.dropna(subset=['fecha'], inplace=True)
        return df[['fecha', 'consumo_kwh']]
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_weather_data(file_path):
    try:
        # Simple loader assuming standard NASA format or cleaned CSV
        df = pd.read_csv(file_path, sep=None, engine='python')
        
        # If it's the raw NASA file with header issues
        if "YEAR" in df.columns and "MO" in df.columns:
             df['fecha'] = pd.to_datetime(df[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
             if 'T2M' in df.columns: df.rename(columns={'T2M': 'temperatura_c'}, inplace=True)
        elif 'fecha' in df.columns:
             df['fecha'] = pd.to_datetime(df['fecha'])
        
        if 'temperatura_c' not in df.columns:
             # Try fallback
             cols = [c for c in df.columns if 'temp' in c.lower()]
             if cols: df.rename(columns={cols[0]: 'temperatura_c'}, inplace=True)
             
        return df[['fecha', 'temperatura_c']]
    except:
        return pd.DataFrame()

# ==========================================
# 3. MAIN APP
# ==========================================

def run():
    # --- PATH SETUP ---
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(os.path.dirname(current_dir)) 
    data_folder = os.path.join(root_dir, 'data')

    st.sidebar.title("🔌 Desglose NILM")
    
    # --- FILE SELECTION ---
    st.sidebar.markdown("### 📂 Cargar Datos")
    
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        if files:
            f_energy = st.sidebar.selectbox("Archivo Consumo (Horario)", files, index=0)
            f_weather = st.sidebar.selectbox("Archivo Clima", [None] + files, index=len(files)-1 if len(files)>1 else 0)
            
            df_consumo = load_hourly_data(os.path.join(data_folder, f_energy))
            if f_weather:
                df_clima = load_weather_data(os.path.join(data_folder, f_weather))
            else:
                df_clima = pd.DataFrame()
        else:
            st.error("No hay archivos CSV en 'data/'")
            return
    else:
        st.error("'data/' folder missing.")
        return

    # --- UI & LOGIC ---
    if df_consumo.empty:
        st.warning("El archivo de consumo no se pudo leer o está vacío.")
        return

    # Merge Data
    if not df_clima.empty:
        try:
            df_merged = pd.merge(df_consumo, df_clima, on='fecha', how='inner')
        except:
             st.warning("Error al unir consumo y clima. Usando solo consumo.")
             df_merged = df_consumo.copy()
             df_merged['temperatura_c'] = 20.0 # Default temp
    else:
        df_merged = df_consumo.copy()
        df_merged['temperatura_c'] = 20.0

    if df_merged.empty:
        st.error("Error: No hay datos coincidentes entre consumo y clima.")
        return

    # --- SIDEBAR CONTROLS (Your Original Logic) ---
    with st.sidebar:
        st.divider()
        st.header("🎛️ Simulación")
        
        # Month Filter
        available_months = df_merged['fecha'].dt.month_name().unique()
        months_to_remove = st.multiselect("Excluir Meses", options=available_months)
        
        if months_to_remove:
            mask_month = ~df_merged['fecha'].dt.month_name().isin(months_to_remove)
            df_merged = df_merged[mask_month]

        # Day Type
        day_type = st.radio("Tipo de Día", ["Laborable", "Fin de Semana"], horizontal=True)
        is_weekday = (day_type == "Laborable")
        mask_day = df_merged['fecha'].dt.dayofweek < 5 if is_weekday else df_merged['fecha'].dt.dayofweek >= 5
        df_filtered = df_merged[mask_day].copy()
        
        if df_filtered.empty: 
            st.warning("No hay datos para esta selección.")
            return
        
        # PARAMETERS
        with st.expander("1. Base y Ventilación", expanded=True):
            base_kw = st.number_input("Base Load [kW]", 0.0, 10000.0, float(df_filtered['consumo_kwh'].min()), step=1.0)
            vent_kw = st.number_input("Vent kW", 0.0, 10000.0, 30.0, step=1.0)
            c1, c2 = st.columns(2)
            v_s, v_e = c1.slider("Horario Vent", 0, 24, (6, 20))
            v_ru = c2.number_input("Ramp Up (h)", 0.0, 5.0, 0.5)
            v_rd = c2.number_input("Ramp Down (h)", 0.0, 5.0, 0.5, key="v_rd")

        with st.expander("2. Iluminación", expanded=False):
            light_kw = st.number_input("Light kW", 0.0, 10000.0, 20.0, step=1.0)
            l_fac = st.slider("Factor Uso %", 0.0, 1.0, 0.8)
            l_sec = st.slider("Seguridad %", 0.0, 0.5, 0.1)
            c3, c4 = st.columns(2)
            l_s, l_e = c3.slider("Horario Luz", 0, 24, (7, 21))
            l_ru = c4.number_input("L-Ramp Up", 0.0, 5.0, 0.5)
            l_rd = c4.number_input("L-Ramp Down", 0.0, 5.0, 0.5)

        with st.expander("3. Climatización (HVAC)", expanded=False):
            therm_kw = st.number_input("Chiller kW", 0.0, 50000.0, 45.0, step=5.0)
            t_s, t_e = st.slider("Horario Clima", 0, 24, (8, 19))
            mode = st.selectbox("Modo", ["Constante", "Depende de Clima"])
            sens, sc, sh = 5.0, 24, 20
            if mode == "Depende de Clima":
                sens = st.slider("Sensibilidad", 1.0, 20.0, 5.0)
                sc = st.number_input("Set Frío", 18, 30, 24)
                sh = st.number_input("Set Calor", 15, 25, 20)
            # Map mode back to English for logic engine if needed, or update engine
            eng_mode = "Weather Driven" if mode == "Depende de Clima" else "Constant"

        with st.expander("4. Procesos Extra", expanded=False):
            if 'n_proc' not in st.session_state: st.session_state['n_proc'] = 0
            b1, b2 = st.columns(2)
            if b1.button("➕"): st.session_state['n_proc'] += 1
            if b2.button("➖"): st.session_state['n_proc'] = max(0, st.session_state['n_proc'] - 1)
            
            procs = []
            for i in range(st.session_state['n_proc']):
                st.markdown(f"**Proc {i+1}**")
                pk = st.number_input(f"kW {i+1}", 0.0, 5000.0, 10.0, key=f"pk{i}")
                ps, pe = st.slider(f"Horario {i+1}", 0, 24, (8, 17), key=f"pt{i}")
                pr_u = st.number_input(f"R-Up {i+1}", 0.0, 5.0, 0.5, key=f"pru{i}")
                pr_d = st.number_input(f"R-Dn {i+1}", 0.0, 5.0, 0.5, key=f"prd{i}")
                has_dip = st.checkbox(f"Parada Comida? {i+1}", value=(i==0), key=f"pd{i}")
                dips = [{'hour': 14, 'factor': 0.5}] if has_dip else []
                procs.append({'name': f"P{i+1}", 'kw': pk, 's': ps, 'e': pe, 'ru': pr_u, 'rd': pr_d, 'dips': dips})
                st.divider()

    # --- CALCULATION ---
    # Create Average Day Profile
    df_avg = df_filtered.groupby(df_filtered['fecha'].dt.hour).agg({
        'consumo_kwh': 'mean', 'temperatura_c': 'mean'
    }).reset_index().rename(columns={'fecha': 'hora'})

    config = {
        'base_kw': base_kw,
        'vent_kw': vent_kw, 'vent_s': v_s, 'vent_e': v_e, 'vent_ru': v_ru, 'vent_rd': v_rd,
        'light_kw': light_kw, 'light_s': l_s, 'light_e': l_e, 'light_fac': l_fac, 'light_sec': l_sec, 'light_ru': l_ru, 'light_rd': l_rd,
        'therm_kw': therm_kw, 'therm_s': t_s, 'therm_e': t_e, 'hvac_mode': eng_mode, 'therm_sens': sens, 'set_c': sc, 'set_h': sh,
        'processes': procs
    }
    
    df_sim = run_simulation(df_avg, config)

    # --- DASHBOARD ---
    st.subheader(f"📊 Gemelo Digital: {day_type}")
    
    # Metrics
    real = df_sim['consumo_kwh'].sum()
    sim = df_sim['sim_total'].sum()
    rmse = np.sqrt(((df_sim['consumo_kwh'] - df_sim['sim_total']) ** 2).mean())
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Real (Día Promedio)", f"{real:,.0f} kWh")
    c2.metric("Simulado", f"{sim:,.0f} kWh", delta=f"{sim-real:,.0f} kWh")
    c3.metric("RMSE (Error Ajuste)", f"{rmse:.2f}", help="Menor es mejor")

    # Chart
    fig = go.Figure()
    x = df_sim['hora']
    layers = [
        ('sim_base', 'Base', 'gray'),
        ('sim_vent', 'Ventilación', '#3498db'),
        ('sim_therm', 'Climatización', '#e74c3c'),
        ('sim_light', 'Iluminación', '#f1c40f'),
        ('sim_proc', 'Procesos', '#e67e22')
    ]
    for col, name, color in layers:
        fig.add_trace(go.Scatter(x=x, y=df_sim[col], stackgroup='one', name=name, line=dict(width=0, color=color)))
        
    fig.add_trace(go.Scatter(x=x, y=df_sim['consumo_kwh'], mode='lines+markers', name='REAL', line=dict(color='black', width=3)))
    fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

    # Split View
    c_left, c_right = st.columns(2)
    with c_left:
        sums = df_sim[['sim_base', 'sim_vent', 'sim_therm', 'sim_light', 'sim_proc']].sum().reset_index()
        sums.columns = ['Category', 'kWh']
        sums['Category'] = sums['Category'].str.replace('sim_', '').str.capitalize()
        st.plotly_chart(px.pie(sums, values='kWh', names='Category', hole=0.4), use_container_width=True)
    
    with c_right:
        df_sim['color'] = np.where(df_sim['diff'] > 0, '#ef5350', '#42a5f5')
        st.plotly_chart(px.bar(df_sim, x='hora', y='diff', title="Diferencia (Real - Sim)"), use_container_width=True)

if __name__ == "__main__":
    run()
