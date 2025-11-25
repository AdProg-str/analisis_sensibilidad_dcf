"""
Aplicación Streamlit para Valuación DCF (FCFF) y Análisis de Sensibilidad (WACC y g).

Estructura general:
1) Carga y procesamiento de archivos (Excel → DataFrames de partidas → FCFF).
2) Visualización de partidas clave y definición de parámetros globales (g, WACC, años).
3) Resultados de valuación por compañía y comparativos básicos.
4) Análisis de sensibilidad: matrices (% y niveles), heatmaps y elasticidades locales.
5) Promedios/desvíos cross-empresa y exportación a Excel de todos los insumos.

Notas metodológicas:
- El valor de la firma se calcula descontando FCFF al WACC y agregando perpetuidad
  creciente con tasa g (donde se asume g < WACC para convergencia).
- El valor por acción = (EV - Deuda Neta) / Acciones en circulación (unidad consistente).
- Las rejillas de WACC/g pueden ser absolutas (bps) o relativas (%).

IMPORTANTE: Este archivo es la UI/flujo de la app. Las transformaciones y la lógica
de valuación están en 'funciones.py', importadas al inicio.
"""

# -----------------------------------
# Imports de librerías y módulos
# -----------------------------------

import streamlit as st                 # Framework web para la app interactiva
import pandas as pd                    # Manejo de tablas y exportaciones
import numpy as np                     # Cálculo numérico
from funciones import (                # Funciones utilitarias del pipeline financiero
    limpiar_excel, extraer_partidas, normalizar_indice, limpiar_partidas,
    get_netdebt, completar_partidas, calcular_nuevas_partidas, calcular_desvios,
    elegir_ultimo_fcff_estable, valuacion_DCF, calcular_waccs, calcular_gs, dcf_scenarios,
    dcf_sensitivity_matrix, calcular_waccs2, calcular_gs2, crear_df_con_elasticidades, acciones
)
import io                                   # Buffers en memoria para construir el Excel de salida
from datetime import datetime, timedelta    # Timestamps para auditoría/exportación
import plotly.graph_objects as go           # Gráficos (barras, heatmaps, etc.)
import plotly.express as px                 # API de alto nivel para gráficas comparativas
import yfinance as yf                       # (Opcional) Queda importado si se extiende a precios de mercado

# Comando de ejecución local de referencia:
# py -m streamlit run app.py --server.address localhost --server.port 8501

# --------------------------------------------------------------------
# Configuración general de la app Streamlit (layout y encabezado)
# --------------------------------------------------------------------

st.set_page_config(
    layout="wide"  # Layout ancho para aprovechar espacio en tablas/gráficos
)

# Título y Descripción
st.markdown(
    "<h1 style='color:#f89100ff; font-size:42px;'>📊 Valuación por DCF & Análisis de Sensibilidad</h1>",
    unsafe_allow_html=True
)

st.markdown("Suba las proyecciones para realizar la valuación por DCF y el análisis de sensibilidad.")

# ------------------------------ Lineas 62-68 -------------------------------------------
# Estado de sesión: estructuras para persistir empresas procesadas y escenarios guardados
# ---------------------------------------------------------------------------------------

# Diccionario {file_key: dict(...)} con lo procesado por archivo
if 'processed_companies' not in st.session_state:
    st.session_state.processed_companies = {}

# Lista de escenarios guardados (cada uno con g, WACC, años, etc.)
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []

# ------------------------------ Lineas 77-142 --------------------------------------
# Sección 1: Carga de archivos y validación de tickers
# - Lee excels, limpia y normaliza partidas
# - Calcula deuda neta, completa partidas faltantes y deriva nuevas (WK, NOPAT, FCFF)
# - Determina el punto de corte de FCFF estable y guarda todo en session_state
# -----------------------------------------------------------------------------------

st.header("1. Carga de Datos Financieros")

# Widget para la carga de archivos
uploaded_files = st.file_uploader(
    "Suba uno o más archivos de Excel",
    type=['xlsx', 'xls'],
    accept_multiple_files=True,
    help="Suba archivos de Excel con los estados financieros para el análisis de la empresa."
)

# Procesamiento de los archivos cargados
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name   # Clave estable por nombre de archivo
        
        # Solo se procesan si no fueron ya procesados
        if file_key not in st.session_state.processed_companies:
            try:
                # 1) Lectura del Excel a DataFrame "crudo"
                excel_data = pd.read_excel(uploaded_file)
                
                # 2) Limpieza de cabeceras/columnas y detección de ticker
                excel_limpio, ticker = limpiar_excel(excel_data)
                
                # 3) Validación de ticker contra la lista soportada (acciones dict)
                if ticker not in acciones:
                    st.error(f"❌ El ticker '{ticker}' del archivo '{uploaded_file.name}' no es compatible. Tickers compatibles: {', '.join(acciones.keys())}")
                    continue
                
                # 4) Selección de partidas relevantes y normalización de nombres
                partidas = extraer_partidas(excel_limpio)
                partidas.index = partidas.index.map(normalizar_indice)
                partidas = limpiar_partidas(partidas)
                
                # 5) Cálculo de Deuda Neta (con fallback si no hay Net/Total/ST+LT)
                netdebt = get_netdebt(partidas)
                
                # 6) Validar y derivar partidas operativas (WK, ΔWK, NOPAT, FCFF)
                partidas = completar_partidas(partidas)
                partidas = calcular_nuevas_partidas(partidas)
                
                # 7) Se selecciona el año en el que se establiza el FCFF
                anio_corte, fcff_hasta_corte = elegir_ultimo_fcff_estable(
                    partidas,
                    fila_fcff="FCFF",
                    k_pos=4
                )
                
                free_cash_flows = np.array(fcff_hasta_corte)[1:]
                
                # Se guarda la información procesada para su posterior uso
                st.session_state.processed_companies[file_key] = {
                    'ticker': ticker,
                    'partidas': partidas,
                    'netdebt': netdebt,
                    'free_cash_flows': free_cash_flows,
                    'anio_corte': anio_corte,
                    'default_wacc': acciones[ticker].wacc,
                    'num_years': len(free_cash_flows)
                }
                
                st.success(f"✅ Procesado correctamente: {uploaded_file.name} (Ticker: {ticker})")
                
            except Exception as e:
                # Manejo defensivo de errores de lectura/formato
                st.error(f"❌ Error al procesar {uploaded_file.name}: {str(e)}")

# ------------------------------ Lineas 152-350
# Sección 2: Visualización de partidas procesadas y parámetros de análisis
# - Muestra partidas clave (NOPAT, D&A, CAPEX, WK, FCFF)
# - Setea sliders para g, WACC y años de DCF
# - Permite usar WACC por defecto o uno global
# - Gestión de escenarios (guardar, listar, comparar)
# ------------------------------

if st.session_state.processed_companies:
    fcffs = []  # Mantiene copias de las partidas seleccionadas para reporte

    # (2.1) Explorador de partidas procesadas por archivo/ticker
    with st.expander("📋 Ver Información Financiera Procesada"):
        for file_key, company_data in st.session_state.processed_companies.items():
            st.markdown(f"### {company_data['ticker']} - {file_key}")
            # Subconjunto de partidas clave para DCF (visual)
            df_fcff = company_data['partidas'].loc[['NOPAT',"D&A",'CAPEX','WK','FCFF']]
            st.dataframe(df_fcff, use_container_width=True)
            fcffs.append(df_fcff)  # Para exportación posterior
            st.markdown("---")

    # (2.2) Parámetros globales de valuación (UI)
    st.markdown("<h3 style='color:#f89100ff; font-size:36px;'>2. Parametros de la Valuación</h1>",
        unsafe_allow_html=True)
    
    # Obtiene el valor mínimo y máximo para el slider selector basandose en todas las empresas
    all_waccs = [data['default_wacc'] for data in st.session_state.processed_companies.values()]
    all_num_years = [data['num_years'] for data in st.session_state.processed_companies.values()]
    
    min_wacc = min(all_waccs)
    max_wacc = max(all_waccs)
    avg_wacc = np.mean(all_waccs)
    max_years = max(all_num_years)
    
    # Tres columnas con sliders y estados
    col1, col2, col3 = st.columns(3)
    
    # Selector de tasa g
    with col1:
        g_rate = st.slider(
            "Tasa de Crecimiento a Perpetuidad (g)",
            min_value=0.0,
            max_value=0.10,
            value=0.03,
            step=0.001,
            format="%.3f",
            help="Tasa de crecimiento terminal para el cálculo de la perpetuidad"
        )
        st.caption(f"Tasa seleccionada: {g_rate:.1%}")
    
    # Selector de WACC
    with col2:
        wacc_override = st.slider(
            "WACC",
            min_value=0.01,
            max_value=0.25,
            value=float(avg_wacc),
            step=0.001,
            format="%.3f",
            help="Costo Promedio Ponderado de Capital (se utilizará el valor predeterminado de la empresa si no se modifica)."
        )
        st.caption(f"WACC seleccionado: {wacc_override:.1%}")
    
    # Cantidad de años de los FCFF a usar en la valuación
    with col3:
        num_years_to_use = st.slider(
            "Número de Años para el DCF",
            min_value=1,
            max_value=max_years,
            value=max_years,
            step=1,
            help="Número de años de flujos de efectivo a usar en la valuación"
        )
        st.caption(f"Usando {num_years_to_use} años")
    
    # Checkbox para usar el WACC predeterminado para cada compañía
    use_default_wacc = st.checkbox(
        "Usar el WACC predeterminado de cada empresa",
        value=True,
        help="""Si se selecciona esta opción, cada empresa utilizará su propio WACC predeterminado. 
                De lo contrario, se utilizará el valor de WACC predeterminado para todas las empresas."""
    )
    
# --------------------------------------------------------------------------
# Gestión de escenarios: guardado, tabla de escenarios y comparación gráfica
# --------------------------------------------------------------------------

    st.subheader("💾 Gestor de Escenarios")
    
    col_scenario1, col_scenario2 = st.columns([2, 1])
    
    # Selector de nombre de escenario
    with col_scenario1:
        # Nombre libre para identificar el escenario
        scenario_name = st.text_input(
            "Nombre del Escenario",
            placeholder="e.g., Base, Optimista, Conservador",
            help="Asignale al escenario un nombre descriptivo"
        )
    
    # Botón de guardado del escenario actual
    with col_scenario2:
        # Botón de guardado: persiste parámetros actuales
        if st.button("💾 Guardar el Escenario Actual", type="primary", use_container_width=True):
            if scenario_name:
                scenario_data = {
                    'Nombre': scenario_name,
                    'Tasa g': g_rate,
                    'WACC': wacc_override,
                    'Años': num_years_to_use,
                    'Usar WACC default': use_default_wacc,
                    'Fecha y Hora': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.saved_scenarios.append(scenario_data)
                st.success(f"✅ Escenario '{scenario_name}' guardado!")
            else:
                st.error("Por favor ingresa un nombre para el escenario")
    
    # Muestra los escenarios guardados
    if st.session_state.saved_scenarios:
        with st.expander(f"📊 Ver y Comparar los Escenarios ({len(st.session_state.saved_scenarios)})", expanded=False):
            # Muestra la tabla con escenarios
            scenario_df = pd.DataFrame(st.session_state.saved_scenarios)
            scenario_display = scenario_df.copy()
            scenario_display['Tasa g'] = scenario_display['Tasa g'].apply(lambda x: f"{x:.2%}")
            scenario_display['WACC'] = scenario_display['WACC'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                scenario_display[['Nombre', 'Tasa g', 'WACC', 'Años', 'Usar WACC default', 'Fecha y Hora']],
                use_container_width=True
            )
            
            # Comparación de escenarios
            st.subheader("Compare Scenarios")
            selected_scenarios = st.multiselect(
                "Seleccione los escenarios para comparar",
                options=[s['Nombre'] for s in st.session_state.saved_scenarios],
                help="Seleccione 2 o más escenarios para comparar los resultados"
            )
            
            if len(selected_scenarios) >= 2:
                # Calcula la valuación para cada escenario seleccionado
                comparison_data = []
                
                for scenario_name in selected_scenarios:
                    scenario = next(s for s in st.session_state.saved_scenarios if s['Nombre'] == scenario_name)
                    
                    # Calcular las valoraciones de cada empresa con los parámetros de este escenario
                    for file_key, company_data in st.session_state.processed_companies.items():
                        ticker = company_data['ticker']
                        
                        wacc_to_use = company_data['default_wacc'] if scenario['Usar WACC default'] else scenario['wacc_override']
                        fcf_to_use = company_data['free_cash_flows'][:scenario['Años']]
                        
                        stock_value = valuacion_DCF(
                            wacc=wacc_to_use,
                            g=scenario['Tasa g'],
                            free_cash_flows=fcf_to_use,
                            ticker=ticker,
                            netdebt=company_data['netdebt']
                        )[0]
                        
                        valor_terminal = valuacion_DCF(
                            wacc=wacc_to_use,
                            g=scenario['Tasa g'],
                            free_cash_flows=fcf_to_use,
                            ticker=ticker,
                            netdebt=company_data['netdebt']
                        )[1]
                        
                        comparison_data.append({
                            'Escenario': scenario_name,
                            'Ticker': ticker,
                            'Valor de la Acción ($)': stock_value,
                            'Peso del Valor Terminal': f"{valor_terminal:.2%}",
                            'g': f"{scenario['Tasa g']:.2%}",
                            'WACC': f"{wacc_to_use:.2%}"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Crea un gráfico comparativo
                fig_comparison = px.bar(
                    comparison_df,
                    x='Ticker',
                    y='Valor de la Acción ($)',
                    color='Escenario',
                    barmode='group',
                    title='Comparación de Escenarios - Valuación',
                    text='Valor por Acción ($)'
                )
                
                fig_comparison.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                fig_comparison.update_layout(height=500)
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Muestra la tabla comparativa de escenarios
                st.dataframe(comparison_df, use_container_width=True)
            
            col_clear, col_space = st.columns([1, 3])
            with col_clear:
                if st.button("🗑️ Borrar Escenarios"):
                    st.session_state.saved_scenarios = []
                    st.rerun()
    
    st.divider()
    
# ------------------------------ 358-418 --------------------------------
# Sección 3: Resultados de valuación DCF por compañía
# - Calcula y muestra métricas (valor estimado, WACC/g usados, años, TV%)
# - Renderiza comparativas simples
# -----------------------------------------------------------------------

    st.markdown(
    "<h3 style='color:#f89100ff; font-size:35px;'>3. Resultados de la Valuación</h1>",
    unsafe_allow_html=True)

    # Guarda la valuación de cada compañía
    company_valuations = {}
    
    # Creación de columnas para mostrar los resultados
    num_companies = len(st.session_state.processed_companies)
    cols = st.columns(min(3, num_companies))
    
    for idx, (file_key, company_data) in enumerate(st.session_state.processed_companies.items()):
        ticker = company_data['ticker']
        
        # Determina que WACC usar
        wacc_to_use = company_data['default_wacc'] if use_default_wacc else wacc_override
        
        # Recorta los cashflows en función del año en que se estabilizan
        fcf_to_use = company_data['free_cash_flows'][:num_years_to_use]
        
        # Cálculo de la valuación
        stock_value = valuacion_DCF(
            wacc=wacc_to_use,
            g=g_rate,
            free_cash_flows=fcf_to_use,
            ticker=ticker,
            netdebt=company_data['netdebt']
        )[0]
        
        valor_terminal = valuacion_DCF(
            wacc=wacc_to_use,
            g=g_rate,
            free_cash_flows=fcf_to_use,
            ticker=ticker,
            netdebt=company_data['netdebt']
        )[1]
        
        company_valuations[ticker] = {
            'stock_value': stock_value,
            'valor_terminal': valor_terminal,
            'wacc_used': wacc_to_use,
            'g_used': g_rate,
            'years_used': num_years_to_use,
            'fcf': fcf_to_use,
            'netdebt': company_data['netdebt'],
        }
        
        # Muestra los resultados en columnas
        col_idx = idx % 3
        with cols[col_idx]:
            st.metric(
                label=f"{ticker}",
                value=f"${stock_value:.2f}",
                delta=None
            )
            st.caption(f"WACC: {wacc_to_use:.2%}")
            st.caption(f"Tasa g: {g_rate:.2%}")
            st.caption(f"Peso del Valor Terminal: {valor_terminal:.2%}")
            st.caption(f"Años: {num_years_to_use}")
    
    st.divider()
    
# -------------------- Lineas 425-492 --------------------------
# Sección 4: Comparativos entre compañías
# - Barras de valor, barras de WACC, línea de valores combinados
# --------------------------------------------------------------

    if len(company_valuations) > 1:
        st.markdown(
        "<h3 style='color:#f89100ff; font-size:36px;'>4. Comparativo</h1>",
        unsafe_allow_html=True)
        
        # Preparación de los datos para los gráficos
        tickers_list = list(company_valuations.keys())
        stock_values = [val['stock_value'] for val in company_valuations.values()]
        waccs_used = [val['wacc_used'] for val in company_valuations.values()]
        gs_used = [val['g_used'] for val in company_valuations.values()]
        
        col_chart1, col_chart2 = st.columns(2)
        
        # Comparación de valores intrínsecos
        with col_chart1:
            fig_value = go.Figure(data=[
                go.Bar(
                    x=tickers_list,
                    y=stock_values,
                    text=[f'${v:.2f}' for v in stock_values],
                    textposition='auto',
                    marker=dict(
                        color=stock_values,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Valor ($)")
                    )
                )
            ])
            
            fig_value.update_layout(
                title="Valor de la Acción",
                xaxis_title="Compañía",
                yaxis_title="Valor de la Acción ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_value, use_container_width=True)
        
        # Comparación de WACCs
        with col_chart2:
            fig_wacc = go.Figure(data=[
                go.Bar(
                    x=tickers_list,
                    y=[w * 100 for w in waccs_used],
                    text=[f'{w:.2%}' for w in waccs_used],
                    textposition='auto',
                    marker=dict(
                        color=waccs_used,
                        colorscale='Blues',
                        showscale=True,
                        colorbar=dict(title="WACC")
                    )
                )
            ])
            
            fig_wacc.update_layout(
                title="Comparación de WACC",
                xaxis_title="Compañía",
                yaxis_title="WACC (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_wacc, use_container_width=True)
        
        st.divider()
    
# -------------------------Lineas 501-666 ----------------------------
# Sección 5: Análisis de sensibilidad
# - Configura rangos y pasos para WACC y g (absolutos o relativos)
# - Genera matrices de sensibilidad (cambios %) y de precios (niveles)
# - Muestra tabla/heatmap y calcula elasticidades locales
# --------------------------------------------------------------------

    st.markdown(
    "<h3 style='color:#f89100ff; font-size:36px;'>5. Análisis de Sensibilidad</h1>",
    unsafe_allow_html=True)

    # Parámetros para la matriz de sensibilidad
    st.subheader("Parámetros de la Matriz de Sensibilidad")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    absolute_changes = st.checkbox('Usar cambios absolutos', value=True)

    # Magnitud de cambio en WACC (ui en %, convertido a fracción)
    with col_s1:
        wacc_sensitivity_pct = st.slider(
            "Variación del WACC",
            min_value=0.5,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="Cambios aplicables al valor base del WACC"
        )
        wacc_sensitivity_range = wacc_sensitivity_pct / 100

        # Leyenda contextual según el modo elegido
        if absolute_changes:    
            st.caption(f'Cambio de {wacc_sensitivity_pct*100} BPS')
        else:
            st.caption(f'Variaciones de un {wacc_sensitivity_pct}%')

    # Magnitud de cambio en g (ui en %, convertido a fracción)
    with col_s2:
        g_sensitivity_pct = st.slider(
            "Variación en la tasa g",
            min_value=0.5,
            max_value=10.0,
            value=0.5,
            step=0.1,
            help="Cambios aplicables al valor base de la g"
        )
        g_sensitivity_range = g_sensitivity_pct / 100
        
        if absolute_changes:    
            st.caption(f'Cambio de {g_sensitivity_pct*100} BPS')
        else:
            st.caption(f'Variaciones de un {g_sensitivity_pct}%')
        
    with col_s3:
        # Cantidad de pasos a cada lado del centro (n) → 2n+1 puntos
        n_steps = st.slider("Número de pasos",
                            min_value=1,
                            max_value=30,
                            value=4,
                            step=1,
                            help="Número de pasos en torno al valor central de WACC/g")
    
    # Parte en la que se generan las matrices de sensibilidad
    st.subheader("Análisis de Sensibilidad por Compañía")

    st.markdown(f"*Se muestran las variaciones porcentuales en torno al escenario base*")
    
    # Seleccionador de tipo de visualización
    viz_type = st.radio(
        "Tipo de Visualización:",
        ["Tabla", "Mapa de Calor", "Ambos"],
        horizontal=True,
        help="Elegi como mostrar el análisis de sensibilidad"
    )

    # Toggle para mostrar matriz de precios en vez de % (tabla)
    boton_precios = st.toggle('Mostrar Precios')
    
    all_sensitivity_matrices = []  # Guarda matrices % por firma
    all_elasticidades = []         # Guarda tablas de elasticidades por firma
    
    for ticker, valuation_data in company_valuations.items():
        
        with st.expander(f"#### {ticker}"):
            # WACC y g central para el ticker
            base_wacc = valuation_data['wacc_used']
            base_g = valuation_data['g_used']
            
            # Selección de generador de array con valores de g y WACC según "absoluto" o "relativo"
            if absolute_changes:
                wacc_values = calcular_waccs2(base_wacc, cambio=wacc_sensitivity_range, n=n_steps)
                g_values = calcular_gs2(base_g, cambio=g_sensitivity_range, n=n_steps)
                
            else:
                wacc_values = calcular_waccs(base_wacc, cambio=wacc_sensitivity_range, n=n_steps)
                g_values = calcular_gs(base_g, cambio=g_sensitivity_range, n=n_steps)        
            
            # Genera la matriz de sensibilidad
            sensitivity_matrix = dcf_sensitivity_matrix(
            wacc_values=wacc_values,
            g_values=g_values,
            free_cash_flows=valuation_data['fcf'],
            ticker=ticker,
            netdebt=valuation_data['netdebt'],
            g_to_use=g_rate
            )

            # Genera la matriz de precios
            prices_matrix = dcf_scenarios(
            wacc_values=wacc_values,
            g_values=g_values,
            free_cash_flows=valuation_data['fcf'],
            ticker=ticker,
            netdebt=valuation_data['netdebt'],
            g_to_use=g_rate
            )

            # Elasticidades locales (fila/columna centrales)
            elasticidades_df = crear_df_con_elasticidades(sensitivity_matrix)[0]
            valores_wacc = crear_df_con_elasticidades(sensitivity_matrix)[1]
            valores_g = crear_df_con_elasticidades(sensitivity_matrix)[2]
        
            # Acumuladores globales      
            all_sensitivity_matrices.append(sensitivity_matrix)
            all_elasticidades.append(elasticidades_df)
            
            # Muestra los resultados en base a la opcion de visualización seleccionada
            if viz_type in ["Tabla", "Ambos"]:
                if not boton_precios: 
                    st.dataframe(
                        sensitivity_matrix.style.format("{:.2f}%"),
                        use_container_width=True
                    )
                else:
                    st.dataframe(
                        prices_matrix,
                        use_container_width=True
                    )
                    
                st.dataframe(elasticidades_df.style.format({
                            'Cambio en BPS WACC': '{:.2f}',
                            'Cambio en BPS g': '{:.2f}',
                            'Elasticidades WACC': '{:.2f}%',
                            'Elasticidades g': '{:.2f}%'
                            }),
                            use_container_width=True)
            
            if viz_type in ["Mapa de Calor", "Ambos"]:
                # Crea un mapa de calor interactivo
                fig = go.Figure(data=go.Heatmap(
                    z=sensitivity_matrix.values,
                    x=sensitivity_matrix.columns,
                    y=sensitivity_matrix.index,
                    colorscale='RdYlGn',
                    text=sensitivity_matrix.values,
                    texttemplate='%{text:.2f}%',
                    textfont={"size": 10},
                    colorbar=dict(title="Variación %"),
                    hoverongaps=False,
                    hovertemplate='WACC: %{x}<br>g: %{y}<br>Variación: %{z:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Mapa de Calor de Sensibilidad",
                    xaxis_title="WACC (%)",
                    yaxis_title="Tasa g (%)",
                    height=500,
                    font=dict(size=11)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
# ------------------------------ Lineas 672-704 ---------------------------------
# Agregados cross-empresa: promedio y desvío estándar de matrices y elasticidades
# -------------------------------------------------------------------------------
    
    if len(all_sensitivity_matrices) > 1 and absolute_changes:
        # Ejes comunes estimados desde los últimos valores calculados
        columnas = valores_wacc
        indice = valores_g

        # Promedio cell-by-cell de las matrices individuales
        average_matrix = pd.DataFrame(sum([df.to_numpy() for df in all_sensitivity_matrices]) 
                                      / len(all_sensitivity_matrices), columns=np.round(columnas), index=indice)
        
        st.subheader("Matriz de Sensibilidades Promedio")
        
        st.dataframe(average_matrix)

        # Desvío estándar muestral por celda
        st.markdown("*Desviación Estándar*")
        desvio_matrix = pd.DataFrame(calcular_desvios(all_sensitivity_matrices), index=average_matrix.index, columns=average_matrix.columns)
        st.dataframe(desvio_matrix)
        
    if len(all_sensitivity_matrices) > 1:  
        # Promedio de elasticidades (columna a columna)
        elasti_prom = pd.DataFrame(sum([df.to_numpy() for df in all_elasticidades]) / len(all_elasticidades), 
                                   columns=["Cambio Relativo WACC (%)", "Elasticidades WACC Promedio", 
                                            "Cambio Relativo g (%)", "Elasticidades g Promedio"])
        # Reemplazo de columnas de cambios relativos por valores comunes
        elasti_prom.iloc[:,0] = crear_df_con_elasticidades(sensitivity_matrix)[4]
        elasti_prom.iloc[:,2] = crear_df_con_elasticidades(sensitivity_matrix)[3]
        
        # Desvío estándar de elasticidades (mismo helper, sobre tabla 2D)
        desvio_elast = pd.DataFrame(calcular_desvios(all_elasticidades), columns=elasti_prom.columns)
        
        st.subheader("Elasticidades Promedio")
        
        st.dataframe(elasti_prom)
        
# ------------------------------ Lineas 712-816 ------------------------
# Sección 6: Exportación a Excel
# - Genera un reporte con resumen de valuaciones, matrices individuales,
#   promedios/desvíos y tabla de elasticidades
# ----------------------------------------------------------------------

    st.markdown(
    "<h3 style='color:#f89100ff; font-size:36px;'>6. Exportar Resultados</h1>",
    unsafe_allow_html=True)
    
    def create_excel_report(company_valuations, all_sensitivity_matrices, avg_matrix=None):
        """Crea un archivo Excel con todos los resultados del análisis"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja resumen con todas las valuciones
            summary_data = []
            for ticker, valuation_data in company_valuations.items():
                summary_data.append({
                    'Ticker': ticker,
                    'Valor de la Acción ($)': valuation_data['stock_value'],
                    'WACC Usado': valuation_data['wacc_used'],
                    'Tasa g': valuation_data['g_used'],
                    'Años Usados': valuation_data['years_used'],
                    'Valor Terminal %': valuation_data['valor_terminal'], 
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
            
            # FCFFs por empresa
            start_row = 1
            
            for fcff in fcffs:
                fcff.to_excel(writer, sheet_name='FCFFs', startrow=start_row)
                start_row += len(fcff.iloc[1]) + 1
            
            # Hojas con matrices por compañía y elasticidades
            sheet_name = 'All sensitivities'
            start_row = 0
            
            # Matrices de Sensibilidad Individuales
            for idx, (ticker, valuation_data) in enumerate(company_valuations.items()):
                sensitivity_matrix = all_sensitivity_matrices[idx]
                elasticidad_matrix = all_elasticidades[idx]
                
                title_df = pd.DataFrame([[ticker]])
                title_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=False)
                title_df.to_excel(writer, sheet_name='Elasticidades', startrow=start_row, index=False, header=False)
                
                sensitivity_matrix.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1)
                elasticidad_matrix.to_excel(writer, sheet_name='Elasticidades', startrow=start_row + 1)
                
                start_row += n_steps * 2 + 4
            
            # Promedios y desvíos
            if absolute_changes and len(all_sensitivity_matrices) > 1:
                average_matrix.to_excel(writer, sheet_name='Averages', startrow=1)
                desvio_matrix.to_excel(writer, sheet_name='Std', startrow=1)
                
            if len(all_sensitivity_matrices) > 1:
                elasti_prom.to_excel(writer, sheet_name='Prom Elast', startrow=1)
                desvio_elast.to_excel(writer, sheet_name='Prom Elast', startrow=len(elasti_prom.iloc[:,0])+3)
        
        output.seek(0) # Rebobina el buffer antes de retornarlo
        return output
    
    # Prepara matriz promedio con una grilla común para la exportación
    avg_matrix_for_download = None
    if len(all_sensitivity_matrices) > 1:

        avg_base_wacc_export = np.mean([val['wacc_used'] for val in company_valuations.values()])
        avg_base_g_export = np.mean([val['g_used'] for val in company_valuations.values()])
        
        common_wacc_export = calcular_waccs(avg_base_wacc_export, cambio=wacc_sensitivity_range/5, n=5)
        common_g_export = calcular_gs(avg_base_g_export, cambio=g_sensitivity_range/5, n=5)
        
        common_matrices_export = []
        for ticker, valuation_data in company_valuations.items():
            cm = dcf_sensitivity_matrix(
                wacc_values=common_wacc_export,
                g_values=common_g_export,
                free_cash_flows=valuation_data['fcf'],
                ticker=ticker,
                netdebt=valuation_data['netdebt'],
                g_to_use=g_rate
            )
            common_matrices_export.append(cm)
        
        avg_matrix_for_download = pd.concat(common_matrices_export).groupby(level=0).mean()

    # (6.2) Construye el Excel y expone botón de descarga
    excel_report = create_excel_report(
        company_valuations, 
        all_sensitivity_matrices,
        avg_matrix_for_download
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.download_button(
        label="📥 Descargar Reporte en Excel",
        data=excel_report,
        file_name=f"Análisis_Sensibilidad_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descarga un reporte en excel con todos los resultados"
    )
    
else:
    # Mensaje inicial cuando aún no hay archivos cargados    
    st.info("👆 Suba uno o más archivos de Excel para comenzar el análisis")

# ---------------------Lineas 821-822 ------------------------
# Pie de página con aclaración de alcance y tickers soportados
# ------------------------------------------------------------
st.divider()
st.caption("Herramienta de Valuación por DCF y Análisis de Sensibilidad | Tickers Soportados: " + ", ".join(acciones.keys()))
