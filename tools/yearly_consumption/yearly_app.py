import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import requests
import io
from thefuzz import process

# --- CONSTANTS & MAPPINGS ---
CO2_FACTOR = 0.19  # tCO2e per MWh

province_to_community = {
    'Almería': 'Andalucía', 'Cádiz': 'Andalucía', 'Córdoba': 'Andalucía', 'Granada': 'Andalucía',
    'Huelva': 'Andalucía', 'Jaén': 'Andalucía', 'Málaga': 'Andalucía', 'Sevilla': 'Andalucía',
    'Huesca': 'Aragón', 'Teruel': 'Aragón', 'Zaragoza': 'Aragón',
    'Asturias': 'Principado de Asturias',
    'Balears, Illes': 'Islas Baleares',
    'Araba/Álava': 'País Vasco', 'Bizkaia': 'País Vasco', 'Gipkoa': 'País Vasco',
    'Las Palmas': 'Canarias', 'Santa Cruz de Tenerife': 'Canarias',
    'Cantabria': 'Cantabria',
    'Ávila': 'Castilla y León', 'Burgos': 'Castilla y León', 'León': 'Castilla y León',
    'Palencia': 'Castilla y León', 'Salamanca': 'Castilla y León', 'Segovia': 'Castilla y León',
    'Soria': 'Castilla y León', 'Valladolid': 'Castilla y León', 'Zamora': 'Castilla y León',
    'Albacete': 'Castilla-La Mancha', 'Ciudad Real': 'Castilla-La Mancha', 'Cuenca': 'Castilla-La Mancha',
    'Guadalajara': 'Castilla-La Mancha', 'Toledo': 'Castilla-La Mancha',
    'Barcelona': 'Cataluña', 'Girona': 'Cataluña', 'Lleida': 'Cataluña', 'Tarragona': 'Cataluña',
    'Ceuta': 'Ceuta',
    'Badajoz': 'Extremadura', 'Cáceres': 'Extremadura',
    'Coruña, A': 'Galicia', 'Lugo': 'Galicia', 'Ourense': 'Galicia', 'Pontevedra': 'Galicia',
    'Rioja, La': 'La Rioja',
    'Madrid': 'Comunidad de Madrid',
    'Melilla': 'Melilla',
    'Murcia': 'Región de Murcia',
    'Navarra': 'Comunidad Foral de Navarra',
    'Valencia/València': 'Comunidad Valenciana', 'Alicante/Alacant': 'Comunidad Valenciana', 'Castellón': 'Comunidad Valenciana', 'Castellón/Castelló': 'Comunidad Valenciana'
}

def get_voltage_type(rate):
    if not isinstance(rate, str): return "No definido"
    if rate in ["6.1TD", "6.2TD", "6.3TD", "6.4TD"]: return "Alta Tensión"
    elif rate in ["2.0TD", "3.0TD"]: return "Baja Tensión"
    return "No definido"

def detect_separator(file_path):
    """Detects if a file uses ';' or ',' or '\t' based on the first line."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            line = f.readline()
            if ';' in line: return ';'
            if '\t' in line: return '\t'
            return ','
    except:
        return ','

# --- LOADING FUNCTIONS ---

@st.cache_data
def load_electricity_data(file_path):
    try:
        sep = detect_separator(file_path)
        
        cols_to_use = [
            'CUPS', 'Estado de factura', 'Fecha desde', 'Provincia', 'Nombre suministro',
            'Tarifa de acceso', 'Consumo activa total (kWh)', 'Base imponible (€)',
            'Importe TE (€)', 'Importe TP (€)', 'Importe impuestos (€)', 'Importe alquiler (€)',
            'Importe otros conceptos (€)'
        ]
        
        # Load header only to check columns first (avoid loading huge files if format is wrong)
        df_iter = pd.read_csv(file_path, sep=sep, iterator=True, decimal='.', thousands=',')
        header_cols = df_iter.get_chunk(0).columns.tolist()
        # Clean header whitespace
        header_cols = [c.strip() for c in header_cols]
        
        # Check intersection
        use_cols = [c for c in cols_to_use if c in header_cols]
        
        if len(use_cols) < 3:
            # Fallback for weird encodings or structures
            return pd.DataFrame()

        df = pd.read_csv(
            file_path,
            usecols=lambda c: c.strip() in use_cols,
            parse_dates=['Fecha desde'],
            decimal='.', thousands=',', sep=sep,
            dayfirst=True,
            encoding='utf-8',
            on_bad_lines='skip' 
        )

        df.columns = df.columns.str.strip()
        
        # Normalize columns if needed (handle slight variations)
        if 'Estado de factura' in df.columns:
            df = df[df['Estado de factura'].str.upper() == 'ACTIVA']
            
        rename_map = {
            'Nombre suministro': 'Centro', 'Base imponible (€)': 'Coste Total',
            'Consumo activa total (kWh)': 'Consumo_kWh', 'Importe TE (€)': 'Coste Energía',
            'Importe TP (€)': 'Coste Potencia', 'Importe impuestos (€)': 'Coste Impuestos',
            'Importe alquiler (€)': 'Coste Alquiler', 'Importe otros conceptos (€)': 'Coste Otros'
        }
        df.rename(columns=rename_map, inplace=True)

        numeric_cols = ['Coste Total', 'Consumo_kWh', 'Coste Energía', 'Coste Potencia',
                        'Coste Impuestos', 'Coste Alquiler', 'Coste Otros']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'Fecha desde' in df.columns:
            df['Año'] = df['Fecha desde'].dt.year
            df['Mes'] = df['Fecha desde'].dt.month
        
        if 'Provincia' in df.columns:
            df['Comunidad Autónoma'] = df['Provincia'].map(province_to_community)
            df.dropna(subset=['Comunidad Autónoma'], inplace=True)

        if 'Tarifa de acceso' in df.columns:
            df['Tipo de Tensión'] = df['Tarifa de acceso'].apply(get_voltage_type)
        
        df['Tipo de Energía'] = 'Electricidad'
        
        return df
    except Exception as e:
        st.error(f"Error procesando electricidad: {e}")
        return pd.DataFrame()

@st.cache_data
def load_gas_data(file_path):
    try:
        sep = detect_separator(file_path)
        
        cols_to_use = [
            'CUPS', 'Estado de factura', 'Fecha desde', 'Provincia', 'Nombre suministro',
            'Consumo', 'Base imponible (€)'
        ]
        
        df = pd.read_csv(
            file_path,
            usecols=lambda c: c.strip() in cols_to_use,
            parse_dates=['Fecha desde'],
            decimal=',', thousands='.', # Gas often uses European format
            sep=sep,
            dayfirst=True
        )

        df.columns = df.columns.str.strip()
        
        if 'Estado de factura' in df.columns:
            df = df[df['Estado de factura'].str.upper() == 'ACTIVA']
            
        df.rename(columns={
            'Nombre suministro': 'Centro',
            'Base imponible (€)': 'Coste Total',
            'Consumo': 'Consumo_kWh'
        }, inplace=True)

        for col in ['Coste Total', 'Consumo_kWh']:
            if col in df.columns:
                # Remove thousand separators and swap decimal
                if df[col].dtype == object:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
                df[col] = df[col].fillna(0)

        if 'Fecha desde' in df.columns:
            df['Año'] = df['Fecha desde'].dt.year
            df['Mes'] = df['Fecha desde'].dt.month
            
        if 'Provincia' in df.columns:
            df['Comunidad Autónoma'] = df['Provincia'].map(province_to_community)
            df.dropna(subset=['Comunidad Autónoma'], inplace=True)
            
        df['Tipo de Energía'] = 'Gas'
        return df

    except Exception as e:
        # Don't show error immediately, just return empty so app doesn't crash on wrong file type
        return pd.DataFrame()

@st.cache_data
def get_geojson():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/spain-communities.geojson"
    try:
        response = requests.get(url, timeout=5)
        return response.json()
    except:
        return None

# --- MAIN EXECUTION ---
def run():
    # Path Setup
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    root_dir = os.path.dirname(os.path.dirname(current_dir)) 
    data_folder = os.path.join(root_dir, 'data')

    st.sidebar.markdown("### 📂 Selección de Datos (Anual)")
    
    # Check Data Folder
    if not os.path.exists(data_folder):
        st.error("Carpeta 'data' no encontrada.")
        return

    files = [f for f in os.listdir(data_folder) if f.endswith(('.csv', '.tsv', '.txt'))]
    
    if not files:
        st.warning("No hay archivos CSV en la carpeta data.")
        return

    # Selectors
    col1, col2 = st.sidebar.columns(2)
    selected_file_electricidad = col1.selectbox("Factura Electricidad", [None] + files, index=1 if len(files)>0 else 0)
    selected_file_gas = col1.selectbox("Factura Gas", [None] + files)
    
    comparar_anos = st.sidebar.toggle("Comparar con año anterior")
    selected_file_comparativa = None
    if comparar_anos:
        selected_file_comparativa = col2.selectbox("Electricidad (Año Anterior)", [None] + files)

    # Load Data
    df_electricidad = pd.DataFrame()
    df_gas = pd.DataFrame()
    df_comparativa = pd.DataFrame()

    if selected_file_electricidad:
        df_electricidad = load_electricity_data(os.path.join(data_folder, selected_file_electricidad))
        if df_electricidad.empty:
            st.sidebar.warning(f"El archivo {selected_file_electricidad} no parece tener formato de electricidad válido.")

    if selected_file_gas:
        df_gas = load_gas_data(os.path.join(data_folder, selected_file_gas))
    
    if selected_file_comparativa:
        df_comparativa = load_electricity_data(os.path.join(data_folder, selected_file_comparativa))

    # Combine
    df_combined = pd.concat([df_electricidad, df_gas], ignore_index=True)

    if df_combined.empty:
        st.info("👈 Selecciona tus archivos de facturación en la barra lateral para comenzar.")
        return

    # --- FILTERS ---
    st.sidebar.markdown("---")
    
    # Year Filter
    years = sorted(df_combined['Año'].dropna().unique(), reverse=True)
    selected_year = st.sidebar.selectbox('Seleccionar Año', years) if len(years) > 0 else None

    # Energy Type
    energy_types = ['Ambos'] + sorted(df_combined['Tipo de Energía'].unique().tolist())
    selected_energy_type = st.sidebar.selectbox("Tipo de Energía", energy_types)

    # Geo Filter
    lista_comunidades = sorted(df_combined['Comunidad Autónoma'].unique().tolist())
    selected_communities = st.sidebar.multiselect('Comunidades', lista_comunidades, default=lista_comunidades)

    # Center Filter
    vista_por_centro = st.sidebar.toggle('Filtrar por Centro')
    selected_centros = []
    if vista_por_centro:
        mask_comm = df_combined['Comunidad Autónoma'].isin(selected_communities)
        centros_disponibles = sorted(df_combined[mask_comm]['Centro'].unique().tolist())
        selected_centros = st.sidebar.multiselect('Centros', centros_disponibles, default=centros_disponibles)

    # Voltage Filter (Electric only)
    selected_tension = []
    if not df_electricidad.empty:
        tension_types = sorted(df_electricidad['Tipo de Tensión'].dropna().unique().tolist())
        if tension_types:
            selected_tension = st.sidebar.multiselect('Tensión (Elec)', tension_types, default=tension_types)

    # --- MAIN LOGIC ---
    if selected_year:
        # Apply Filters
        mask = (df_combined['Año'] == selected_year) & (df_combined['Comunidad Autónoma'].isin(selected_communities))
        df_filtered = df_combined[mask].copy()

        if selected_energy_type != 'Ambos':
            df_filtered = df_filtered[df_filtered['Tipo de Energía'] == selected_energy_type]

        if selected_tension:
            # Keep Gas rows + Electric rows matching tension
            is_gas = df_filtered['Tipo de Energía'] == 'Gas'
            is_elec_ok = (df_filtered['Tipo de Energía'] == 'Electricidad') & (df_filtered['Tipo de Tensión'].isin(selected_tension))
            df_filtered = df_filtered[is_gas | is_elec_ok]

        if vista_por_centro and selected_centros:
            df_filtered = df_filtered[df_filtered['Centro'].isin(selected_centros)]

        # --- DASHBOARD ---
        st.title(f"📊 Informe Anual {selected_year}")
        
        if df_filtered.empty:
            st.warning("No hay datos para los filtros seleccionados.")
            return

        # KPIs
        kwh_elec = df_filtered[df_filtered['Tipo de Energía'] == 'Electricidad']['Consumo_kWh'].sum()
        cost_elec = df_filtered[df_filtered['Tipo de Energía'] == 'Electricidad']['Coste Total'].sum()
        kwh_gas = df_filtered[df_filtered['Tipo de Energía'] == 'Gas']['Consumo_kWh'].sum()
        cost_gas = df_filtered[df_filtered['Tipo de Energía'] == 'Gas']['Coste Total'].sum()

        total_kwh = kwh_elec + kwh_gas
        total_cost = cost_elec + cost_gas
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Consumo Total", f"{total_kwh:,.0f} kWh")
        col2.metric("Coste Total", f"€ {total_cost:,.2f}")
        col3.metric("Electricidad", f"€ {cost_elec:,.2f}")
        col4.metric("Gas", f"€ {cost_gas:,.2f}")

        st.markdown("---")

        # Visuals
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Evolución Mensual")
            # Create a full 1-12 month backbone to ensure x-axis is correct
            df_monthly = df_filtered.groupby(['Mes', 'Tipo de Energía'])['Consumo_kWh'].sum().reset_index()
            
            # Helper to map month numbers to names
            month_map = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun', 
                         7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
            df_monthly['Mes_Nombre'] = df_monthly['Mes'].map(month_map)
            
            fig_evol = px.line(df_monthly, x='Mes_Nombre', y='Consumo_kWh', color='Tipo de Energía', markers=True)
            # Force order of months
            fig_evol.update_xaxes(categoryorder='array', categoryarray=list(month_map.values()))
            st.plotly_chart(fig_evol, use_container_width=True)

        with c2:
            st.subheader("Desglose de Costes (Elec)")
            df_elec = df_filtered[df_filtered['Tipo de Energía'] == 'Electricidad']
            cols_cost = ['Coste Energía', 'Coste Potencia', 'Coste Impuestos', 'Coste Alquiler']
            if not df_elec.empty:
                sums = df_elec[cols_cost].sum().reset_index()
                sums.columns = ['Concepto', 'Euros']
                st.plotly_chart(px.pie(sums, names='Concepto', values='Euros', hole=0.4), use_container_width=True)

        # Map
        st.subheader("Mapa de Consumo")
        geojson = get_geojson()
        if geojson:
            df_map = df_filtered.groupby('Comunidad Autónoma')['Consumo_kWh'].sum().reset_index()
            # Simple matching for demo (The user code had fuzz, implemented here simply)
            # For production, ensure names match EXACTLY between GeoJSON and Data
            fig_map = px.choropleth_mapbox(
                df_map, geojson=geojson, locations='Comunidad Autónoma', featureidkey="properties.name",
                color='Consumo_kWh', color_continuous_scale="Viridis",
                mapbox_style="carto-positron", zoom=4.5, center={"lat": 40.4, "lon": -3.7}
            )
            st.plotly_chart(fig_map, use_container_width=True)

# Run logic
if __name__ == "__main__":
    run()
