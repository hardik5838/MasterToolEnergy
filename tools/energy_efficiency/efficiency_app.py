import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Carga y Cacheo de Datos ---
@st.cache_data
def load_data(file_path):
    """Carga, limpia y procesa los datos de la auditoría energética."""
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        # Unifica el renombrado para manejar tanto CSVs en inglés como en español
        df.rename(columns={
            'Center': 'Centro', 'Measure': 'Medida',
            'Energy Saved': 'Ahorro energético', 'Money Saved': 'Ahorro económico',
            'Investment': 'Inversión', 'Pay back period': 'Periodo de retorno',
            'Energía Ahorrada (kWh/año)': 'Ahorro energético', 'Dinero Ahorrado (€/año)': 'Ahorro económico',
            'Inversión (€)': 'Inversión', 'Periodo de Amortización (años)': 'Periodo de retorno'
        }, inplace=True)
        for col in ['Ahorro energético', 'Ahorro económico', 'Inversión', 'Periodo de retorno']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        return pd.DataFrame()

def run():
    # --- PATH SETUP ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    DATA_DIR = os.path.join(root_dir, 'data')

    st.sidebar.title('⚡ Filtros de análisis')

    # --- File Selection Logic ---
    if not os.path.exists(DATA_DIR):
        st.sidebar.error(f"Carpeta 'data' no encontrada en {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        st.sidebar.warning("No se encontraron archivos CSV en la carpeta 'data'.")
        return

    # Try to find '2025' file as default, else first one
    default_index = 0
    for i, f in enumerate(files):
        if "2025" in f:
            default_index = i
            break
            
    selected_file = st.sidebar.selectbox("Seleccionar Auditoría", files, index=default_index)
    file_path = os.path.join(DATA_DIR, selected_file)
    df_original = load_data(file_path)

    if df_original.empty:
        st.warning("El archivo seleccionado está vacío o no tiene el formato correcto.")
        return

    # --- FILTROS LATERALES ---
    tipo_analisis = st.sidebar.radio(
        "Seleccionar Tipo de Análisis",
        ('Tipo de Medida', 'Tipo de Intervención', 'Impacto Financiero', 'Tipo de sistema', 'Tipo de Ahorro Energético')
    )
    
    # Filtro ROI
    filtros_roi = []
    if tipo_analisis == 'Impacto Financiero':
        st.sidebar.markdown("**Filtro de ROI (Financiero)**")
        opciones_roi = [
            'Sin Coste / Inmediato', 
            'Resultados Rápidos (< 2 años)', 
            'Proyectos Estándar (2-5 años)', 
            'Inversiones Estratégicas (> 5 años)'
        ]
        filtros_roi = st.sidebar.multiselect("Rangos de ROI", options=opciones_roi, default=opciones_roi)

    mostrar_porcentaje = st.sidebar.toggle('Mostrar valores en porcentaje')
    st.sidebar.markdown("---")
    
    # Filtro Medidas
    if 'Medida' in df_original.columns:
        todas_medidas = sorted(df_original['Medida'].dropna().unique().tolist())
        medidas_seleccionadas_filtro = st.sidebar.multiselect(
            "Filtrar por Medidas específicas:",
            options=todas_medidas,
            default=todas_medidas,
            placeholder="Selecciona medidas..."
        )
    else:
        medidas_seleccionadas_filtro = []

    st.sidebar.markdown("---")
    vista_detallada = st.sidebar.toggle('Mostrar vista detallada por centro')

    # Filtros Geo/Centro
    if 'Comunidad Autónoma' in df_original.columns:
        lista_comunidades = sorted(df_original['Comunidad Autónoma'].unique().tolist())
        
        # Use session state to persist selection across reruns within the same file
        key_com = f'com_sel_{selected_file}'
        if key_com not in st.session_state:
            st.session_state[key_com] = lista_comunidades
            
        comunidades_seleccionadas = st.sidebar.multiselect(
            'Seleccionar Comunidades', 
            lista_comunidades, 
            default=st.session_state[key_com],
            key=f"multiselect_{key_com}" # Unique key
        )
        
        if st.sidebar.button("Reset Comunidades", use_container_width=True):
             st.session_state[key_com] = lista_comunidades
             st.rerun()

        centros_seleccionados = []
        if vista_detallada and 'Centro' in df_original.columns:
            mask_com = df_original['Comunidad Autónoma'].isin(comunidades_seleccionadas)
            centros_disponibles = sorted(df_original[mask_com]['Centro'].unique().tolist())
            
            st.sidebar.write("Selección de Centros:")
            if st.sidebar.button("Todos los Centros"):
                centros_seleccionados = centros_disponibles
            else:
                centros_seleccionados = st.sidebar.multiselect('Centros', centros_disponibles, default=centros_disponibles)
    else:
        comunidades_seleccionadas = []
        centros_seleccionados = []


    # --- LOGICA DE CATEGORIZACION ---
    # (Copied from user code)
    mapeo_medidas = {
        "Regulación de la temperatura de consigna": {"Category": "Medidas de control térmico", "Code": "A.1"},
        "Sustitución de equipos de climatización": {"Category": "Medidas de control térmico", "Code": "A.2"},
        "Instalación cortina de aire": {"Category": "Medidas de control térmico", "Code": "A.3"},
        "Instalación de temporizador digital": {"Category": "Medidas de control térmico", "Code": "A.4"},
        "Regulación de ventilación mediante sonda de CO2": {"Category": "Medidas de control térmico", "Code": "A.5"},
        "Recuperadores de calor": {"Category": "Medidas de control térmico", "Code": "A.6"},
        "Ajuste O2 en caldera gasóleo C": {"Category": "Medidas de control térmico", "Code": "A.7"},
        "Instalación de Variadores de frecuencia en bombas hidráulicas": {"Category": "Medidas de control térmico", "Code": "A.8"},
        "Instalación Solar térmica": {"Category": "Medidas de control térmico", "Code": "A.9"},
        "Aislamiento Térmico de Tuberías y Redes": {"Category": "Medidas de control térmico", "Code": "A.10"},
        "Mejora de la Eficiencia en Calderas": {"Category": "Medidas de control térmico", "Code": "A.11"},
        "Optimización de la potencia contratada": {"Category": "Medidas de gestión energética", "Code": "B.1"},
        "Sistema de Gestión Energética": {"Category": "Medidas de gestión energética", "Code": "B.2"},
        "Eliminación de la energía reactiva": {"Category": "Medidas de gestión energética", "Code": "B.3"},
        "Reducción del consumo remanente": {"Category": "Medidas de gestión energética", "Code": "B.4"},
        "Promover la cultura energética": {"Category": "Medidas de gestión energética", "Code": "B.5"},
        "Instalación Fotovoltaica": {"Category": "Medidas de gestión energética", "Code": "B.6"},
        "Instalación de Paneles Solares (Fotovoltaicos o Híbridos)": {"Category": "Medidas de gestión energética", "Code": "B.6"},
        "Cambio Iluminacion LED": {"Category": "Medidas de control de iluminación", "Code": "C.1"},
        "Sustitución de luminarias a LED": {"Category": "Medidas de control de iluminación", "Code": "C.1"},
        "Instalación regletas programables": {"Category": "Medidas de control de iluminación", "Code": "C.2"},
        "Mejora en el control de la iluminación": {"Category": "Medidas de control de iluminación", "Code": "C.3"},
        "Mejora en el control actual de iluminación": {"Category": "Medidas de control de iluminación", "Code": "C.3"},
        "Mejora en el control actual": {"Category": "Medidas de control de iluminación", "Code": "C.3"},
        "Sustitución de luminarias a LED y mejora en su control": {"Category": "Medidas de control de iluminación", "Code": "C.4"},
        "Renovación de Equipamiento Específico": {"Category": "Medidas de equipamiento general", "Code": "D.1"}
    }
        
    def categorizar_por_tipo(df_in):
        def get_info(texto_medida):
            for nombre_estandar, info in mapeo_medidas.items():
                if isinstance(texto_medida, str) and nombre_estandar.lower() in texto_medida.lower():
                    return pd.Series([info['Category'], info['Code']])
            return pd.Series(['Sin categorizar', 'Z.Z'])
        df_in[['Categoría', 'Base Código Medida']] = df_in['Medida'].apply(get_info)
        return df_in

    def categorizar_por_intervencion(df_in):
        def get_type(medida):
            if not isinstance(medida, str): return 'Intervenciones Específicas'
            medida = medida.lower()
            if any(word in medida for word in ["instalación", "batería", "recuperadores", "solar", "fotovoltaica"]): return 'Instalación de Nuevos Sistemas'
            if any(word in medida for word in ["sustitución", "cambio", "mejora", "aislamiento"]): return 'Reforma y Actualización de Equipos'
            if any(word in medida for word in ["prácticas", "cultura", "regulación", "optimización", "reducción"]): return 'Operacional y Comportamental'
            return 'Intervenciones Específicas'
        df_in['Categoría'] = df_in['Medida'].apply(get_type)
        return df_in

    def categorizar_por_financiero(df_in):
        def get_type(retorno):
            if pd.isna(retorno): return 'Desconocido'
            if retorno <= 0: return 'Sin Coste / Inmediato'
            if retorno < 2: return 'Resultados Rápidos (< 2 años)'
            if retorno <= 5: return 'Proyectos Estándar (2-5 años)'
            return 'Inversiones Estratégicas (> 5 años)'
        df_in['Categoría'] = df_in['Periodo de retorno'].apply(get_type)
        return df_in

    def categorizar_por_funcion(df_in):
        def get_type(medida):
            if not isinstance(medida, str): return 'Otras Funciones'
            medida = medida.lower()
            if any(word in medida for word in ["hvac", "climatización", "temperatura", "ventilación", "aislamiento", "cortina", "calor", "termo"]): return 'Envolvente y Climatización (HVAC)'
            if any(word in medida for word in ["led", "iluminación", "luminarias", "eléctrico", "potencia", "reactiva", "condensadores", "regletas"]): return 'Iluminación y Electricidad'
            if any(word in medida for word in ["gestión", "fotovoltaica", "solar", "prácticas", "remanente", "cultura"]): return 'Gestión y Estrategia Energética'
            return 'Otras Funciones'
        df_in['Categoría'] = df_in['Medida'].apply(get_type)
        return df_in
        
    def categorizar_por_ahorro_energetico(df_in):
        def get_type(medida):
            if not isinstance(medida, str): return 'Mixto / Operacional'
            medida = medida.lower()
            if any(word in medida for word in ["gasóleo", "diesel", "caldera", "térmica"]): return 'Ahorros Térmicos (Gas/Combustible)'
            if any(word in medida for word in ["led", "iluminación", "fotovoltaica", "eléctrico", "potencia", "reactiva", "variadores", "bombas", "regletas"]): return 'Ahorros Eléctricos'
            return 'Mixto / Operacional'
        df_in['Categoría'] = df_in['Medida'].apply(get_type)
        return df_in

    # --- MAIN LOGIC ---
    
    # Apply Categorization
    mapa_funciones = {
        'Tipo de Medida': categorizar_por_tipo,
        'Tipo de Intervención': categorizar_por_intervencion,
        'Impacto Financiero': categorizar_por_financiero,
        'Tipo de sistema': categorizar_por_funcion,
        'Tipo de Ahorro Energético': categorizar_por_ahorro_energetico,
    }
    
    funcion_a_usar = mapa_funciones.get(tipo_analisis)
    df_categorizado = funcion_a_usar(df_original.copy())
    
    # Apply Filters
    # 1. Measure Filter
    if medidas_seleccionadas_filtro:
        df_categorizado = df_categorizado[df_categorizado['Medida'].isin(medidas_seleccionadas_filtro)]
    
    # 2. ROI Filter
    if tipo_analisis == 'Impacto Financiero' and filtros_roi:
        df_categorizado = df_categorizado[df_categorizado['Categoría'].isin(filtros_roi)]
        
    # 3. Community/Center Filter
    if comunidades_seleccionadas:
        df_filtrado = df_categorizado[df_categorizado['Comunidad Autónoma'].isin(comunidades_seleccionadas)]
        if vista_detallada and centros_seleccionados:
             df_filtrado = df_filtrado[df_filtrado['Centro'].isin(centros_seleccionados)]
    else:
        df_filtrado = pd.DataFrame(columns=df_categorizado.columns)

    # --- DASHBOARD UI ---
    st.title(f"📊 Auditoría: {selected_file}")

    if df_filtrado.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    # KPIs
    inversion = df_filtrado['Inversión'].sum()
    ahorro_eco = df_filtrado['Ahorro económico'].sum()
    ahorro_ener = df_filtrado['Ahorro energético'].sum()
    roi_val = (ahorro_eco / inversion * 100) if inversion > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inversión Total", f"€ {inversion:,.0f}")
    c2.metric("Ahorro Económico", f"€ {ahorro_eco:,.0f}")
    c3.metric("Ahorro Energético", f"{ahorro_ener:,.0f} kWh")
    c4.metric("ROI Estimado", f"{roi_val:.1f} %")
    
    st.markdown("---")
    
    col_agrup = 'Centro' if vista_detallada else 'Comunidad Autónoma'
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución de Medidas")
        df_counts = df_filtrado.groupby([col_agrup, 'Categoría']).size().reset_index(name='Recuento')
        fig1 = px.bar(df_counts, x=col_agrup, y='Recuento', color='Categoría', title=f"Medidas por {col_agrup}")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Ahorro Económico")
        df_eco = df_filtrado.groupby(col_agrup)['Ahorro económico'].sum().reset_index()
        fig2 = px.pie(df_eco, names=col_agrup, values='Ahorro económico', hole=0.4, title=f"Ahorro Económico por {col_agrup}")
        st.plotly_chart(fig2, use_container_width=True)

    # Sankey (Simplified for robustness)
    st.subheader("Flujo: Categoría -> Destino")
    if not df_filtrado.empty and inversion > 0:
        df_sankey = df_filtrado.groupby(['Categoría', col_agrup])['Inversión'].sum().reset_index()
        # Filter small values for cleaner chart
        df_sankey = df_sankey[df_sankey['Inversión'] > 0]
        
        if not df_sankey.empty:
            cats = list(df_sankey['Categoría'].unique())
            dests = list(df_sankey[col_agrup].unique())
            nodes = cats + dests
            node_map = {n: i for i, n in enumerate(nodes)}
            
            sources = [node_map[r['Categoría']] for _, r in df_sankey.iterrows()]
            targets = [node_map[r[col_agrup]] for _, r in df_sankey.iterrows()]
            values = df_sankey['Inversión'].tolist()
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(label=nodes, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values)
            )])
            fig_sankey.update_layout(title_text="Flujo de Inversión", height=400)
            st.plotly_chart(fig_sankey, use_container_width=True)

    # Table
    st.subheader("Detalle de Datos")
    st.dataframe(df_filtrado[['Centro', 'Medida', 'Categoría', 'Inversión', 'Ahorro económico', 'Periodo de retorno']], use_container_width=True)

if __name__ == "__main__":
    run()
