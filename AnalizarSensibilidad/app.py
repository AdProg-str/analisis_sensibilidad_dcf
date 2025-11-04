import streamlit as st
import pandas as pd
import numpy as np
from funciones import (
    limpiar_excel, extraer_partidas, normalizar_indice, limpiar_partidas,
    get_netdebt, completar_partidas, calcular_nuevas_partidas, calcular_desvios,
    elegir_ultimo_fcff_estable, valuacion_DCF, calcular_waccs, calcular_gs, dcf_scenarios,
    dcf_sensitivity_matrix, calcular_waccs2, calcular_gs2, crear_df_con_elasticidades, acciones
)
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# py -m streamlit run app.py --server.address localhost --server.port 8501

# Mapeo de tickers para yfinance (algunos requieren formato especial)
def get_yfinance_ticker(ticker):
    """Mapear tickers internos a formato compatible con yfinance"""
    ticker_map = {
        'BRK': 'BRK-B',  # Berkshire Hathaway Clase B
    }
    return ticker_map.get(ticker, ticker)

# Configuración de página
st.set_page_config(
    layout="wide"
)

st.markdown(
    "<h1 style='color:#f89100ff; font-size:42px;'>📊 Valuación DCF y Análisis de Sensibilidad</h1>",
    unsafe_allow_html=True
)

# Título y descripción
st.markdown("Cargá archivos de Excel con datos financieros para realizar valuaciones DCF y análisis de sensibilidad.")

# Inicializar estado de sesión para empresas procesadas
if 'processed_companies' not in st.session_state:
    st.session_state.processed_companies = {}

# Inicializar estado de sesión para escenarios
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []

# Sección de subida de archivos
st.header("1. Cargar datos financieros")
uploaded_files = st.file_uploader(
    "Cargar uno o más archivos de Excel",
    type=['xlsx', 'xls'],
    accept_multiple_files=True,
    help="Subí archivos de Excel con estados financieros para analizar empresas"
)

# Procesar archivos cargados
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        
        # Solo procesar si no fue procesado antes
        if file_key not in st.session_state.processed_companies:
            try:
                # Leer Excel
                excel_data = pd.read_excel(uploaded_file)
                
                # Pipeline de limpieza
                excel_limpio, ticker = limpiar_excel(excel_data)
                
                # Verificar ticker soportado
                if ticker not in acciones:
                    st.error(f"❌ El ticker '{ticker}' del archivo '{uploaded_file.name}' no está soportado. Tickers soportados: {', '.join(acciones.keys())}")
                    continue
                
                # Extraer y procesar partidas
                partidas = extraer_partidas(excel_limpio)
                partidas.index = partidas.index.map(normalizar_indice)
                partidas = limpiar_partidas(partidas)
                
                # Deuda neta
                netdebt = get_netdebt(partidas)
                
                # Completar y calcular nuevas partidas
                partidas = completar_partidas(partidas)
                partidas = calcular_nuevas_partidas(partidas)
                
                # FCFF estable
                anio_corte, fcff_hasta_corte = elegir_ultimo_fcff_estable(
                    partidas,
                    fila_fcff="FCFF",
                    k_pos=4
                )
                
                free_cash_flows = np.array(fcff_hasta_corte)[1:]
                
                # Guardar datos procesados
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
                st.error(f"❌ Error al procesar {uploaded_file.name}: {str(e)}")

# Mostrar análisis si hay empresas procesadas
if st.session_state.processed_companies:
    # Información adicional
    with st.expander("📋 Ver datos financieros procesados"):
        for file_key, company_data in st.session_state.processed_companies.items():
            st.markdown(f"### {company_data['ticker']} - {file_key}")
            st.dataframe(company_data['partidas'].loc[['NOPAT',"D&A",'CAPEX','WK','FCFF']], use_container_width=True)
            st.markdown("---")
    
    st.markdown(
        "<h3 style='color:#f89100ff; font-size:36px;'>2. Parámetros de análisis</h3>",
        unsafe_allow_html=True)
    
    # Rango de sliders según empresas
    all_waccs = [data['default_wacc'] for data in st.session_state.processed_companies.values()]
    all_num_years = [data['num_years'] for data in st.session_state.processed_companies.values()]
    
    min_wacc = min(all_waccs)
    max_wacc = max(all_waccs)
    avg_wacc = np.mean(all_waccs)
    
    max_years = max(all_num_years)
    
    # Tres columnas de parámetros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        g_rate = st.slider(
            "Tasa de crecimiento (g)",
            min_value=0.0,
            max_value=0.10,
            value=0.03,
            step=0.001,
            format="%.3f",
            help="Tasa de crecimiento a perpetuidad para el cálculo del valor terminal"
        )
        st.caption(f"Seleccionado: {g_rate:.1%}")
    
    with col2:
        # WACC por defecto promedio como valor inicial
        wacc_override = st.slider(
            "WACC (sobrescribir)",
            min_value=0.01,
            max_value=0.25,
            value=float(avg_wacc),
            step=0.001,
            format="%.3f",
            help="Costo Promedio Ponderado de Capital (se usará el WACC propio de cada empresa si no se modifica)"
        )
        st.caption(f"Seleccionado: {wacc_override:.1%}")
    
    with col3:
        num_years_to_use = st.slider(
            "Cantidad de años para el DCF",
            min_value=1,
            max_value=max_years,
            value=max_years,
            step=1,
            help="Cantidad de años de flujos a utilizar en la valuación"
        )
        st.caption(f"Usando {num_years_to_use} años")
    
    # Checkbox para usar WACC por defecto por empresa
    use_default_wacc = st.checkbox(
        "Usar el WACC predeterminado de cada empresa",
        value=True,
        help="Si está tildado, cada empresa usa su WACC propio. Si no, se aplica el WACC sobrescrito a todas."
    )
    
    # Gestión de escenarios
    st.subheader("💾 Gestión de escenarios")
    
    col_scenario1, col_scenario2 = st.columns([2, 1])
    
    with col_scenario1:
        scenario_name = st.text_input(
            "Nombre del escenario",
            placeholder="Ej.: Caso base, Optimista, Conservador",
            help="Poné un nombre descriptivo al escenario"
        )
    
    with col_scenario2:
        if st.button("💾 Guardar escenario actual", type="primary", use_container_width=True):
            if scenario_name:
                scenario_data = {
                    'name': scenario_name,
                    'g_rate': g_rate,
                    'wacc_override': wacc_override,
                    'num_years': num_years_to_use,
                    'use_default_wacc': use_default_wacc,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.saved_scenarios.append(scenario_data)
                st.success(f"✅ ¡Escenario '{scenario_name}' guardado!")
            else:
                st.error("Por favor, ingresá un nombre de escenario")
    
    # Mostrar escenarios guardados
    if st.session_state.saved_scenarios:
        with st.expander(f"📊 Ver y comparar escenarios guardados ({len(st.session_state.saved_scenarios)})", expanded=False):
            # Tabla de escenarios
            scenario_df = pd.DataFrame(st.session_state.saved_scenarios)
            scenario_display = scenario_df.copy()
            scenario_display['g_rate'] = scenario_display['g_rate'].apply(lambda x: f"{x:.2%}")
            scenario_display['wacc_override'] = scenario_display['wacc_override'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                scenario_display[['name', 'g_rate', 'wacc_override', 'num_years', 'use_default_wacc', 'timestamp']],
                use_container_width=True
            )
            
            # Comparación de escenarios
            st.subheader("Comparar escenarios")
            selected_scenarios = st.multiselect(
                "Seleccioná los escenarios a comparar",
                options=[s['name'] for s in st.session_state.saved_scenarios],
                help="Elegí 2 o más escenarios para comparar resultados"
            )
            
            if len(selected_scenarios) >= 2:
                # Valuaciones por escenario
                comparison_data = []
                
                for scenario_name in selected_scenarios:
                    scenario = next(s for s in st.session_state.saved_scenarios if s['name'] == scenario_name)
                    
                    # Calcular valuaciones con parámetros del escenario
                    for file_key, company_data in st.session_state.processed_companies.items():
                        ticker = company_data['ticker']
                        
                        wacc_to_use = company_data['default_wacc'] if scenario['use_default_wacc'] else scenario['wacc_override']
                        fcf_to_use = company_data['free_cash_flows'][:scenario['num_years']]
                        
                        stock_value = valuacion_DCF(
                            wacc=wacc_to_use,
                            g=scenario['g_rate'],
                            free_cash_flows=fcf_to_use,
                            ticker=ticker,
                            netdebt=company_data['netdebt']
                        )[0]
                        
                        valor_terminal = valuacion_DCF(
                            wacc=wacc_to_use,
                            g=scenario['g_rate'],
                            free_cash_flows=fcf_to_use,
                            ticker=ticker,
                            netdebt=company_data['netdebt']
                        )[1]
                        
                        comparison_data.append({
                            'Escenario': scenario_name,
                            'Ticker': ticker,
                            'Valor de la acción ($)': stock_value,
                            'Valor terminal': valor_terminal,
                            'g': f"{scenario['g_rate']:.2%}",
                            'WACC': f"{wacc_to_use:.2%}"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Gráfico comparativo
                fig_comparison = px.bar(
                    comparison_df,
                    x='Ticker',
                    y='Valor de la acción ($)',
                    color='Escenario',
                    barmode='group',
                    title='Comparación de escenarios - Valuaciones por acción',
                    text='Valor de la acción ($)'
                )
                
                fig_comparison.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                fig_comparison.update_layout(height=500)
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Tabla comparativa
                st.dataframe(comparison_df, use_container_width=True)
            
            col_clear, col_space = st.columns([1, 3])
            with col_clear:
                if st.button("🗑️ Borrar todos los escenarios"):
                    st.session_state.saved_scenarios = []
                    st.rerun()
    
    st.divider()
    
    # Resultados de valuación
    st.markdown(
    "<h3 style='color:#f89100ff; font-size:35px;'>3. Resultados de valuación DCF</h3>",
    unsafe_allow_html=True)

    # Guardar valuaciones por empresa
    company_valuations = {}
    
    # Columnas para mostrar resultados
    num_companies = len(st.session_state.processed_companies)
    cols = st.columns(min(3, num_companies))
    
    for idx, (file_key, company_data) in enumerate(st.session_state.processed_companies.items()):
        ticker = company_data['ticker']
        
        # WACC a usar
        wacc_to_use = company_data['default_wacc'] if use_default_wacc else wacc_override
        
        # Flujos según años seleccionados
        fcf_to_use = company_data['free_cash_flows'][:num_years_to_use]
        
        # Valuación
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
        
        # Mostrar en columnas
        col_idx = idx % 3
        with cols[col_idx]:
            st.metric(
                label=f"{ticker}",
                value=f"${stock_value:.2f}",
                delta=None
            )
            st.caption(f"WACC: {wacc_to_use:.2%}")
            st.caption(f"Crecimiento g: {g_rate:.2%}")
            st.caption(f"Valor terminal (%): {valor_terminal:.2%}")
            st.caption(f"Años: {num_years_to_use}")
    
    st.divider()
    
    # Gráficos comparativos
    if len(company_valuations) > 1:
        st.markdown(
        "<h3 style='color:#f89100ff; font-size:36px;'>4. Análisis comparativo</h3>",
        unsafe_allow_html=True)
        
        # Datos para gráficos
        tickers_list = list(company_valuations.keys())
        stock_values = [val['stock_value'] for val in company_valuations.values()]
        waccs_used = [val['wacc_used'] for val in company_valuations.values()]
        gs_used = [val['g_used'] for val in company_valuations.values()]
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Comparación de valor por acción
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
                title="Comparación de valor por acción",
                xaxis_title="Empresa",
                yaxis_title="Valor ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_value, use_container_width=True)
        
        with col_chart2:
            # Comparación de WACC
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
                xaxis_title="Empresa",
                yaxis_title="WACC (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_wacc, use_container_width=True)
        
        # Métricas combinadas
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Scatter(
            x=tickers_list,
            y=stock_values,
            name='Valor por acción',
            mode='lines+markers',
            yaxis='y',
            line=dict(width=3),
            marker=dict(size=10)
        ))
        
        fig_combined.update_layout(
            title="Métricas de valuación entre empresas",
            xaxis_title="Empresa",
            yaxis_title="Valor ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        st.divider()
    
    # Precio histórico vs valor intrínseco
    st.header("4.5. Precio histórico vs valor intrínseco")
    
    with st.expander("📈 Ver comparación de precio histórico", expanded=False):
        st.markdown("Compará los valores intrínsecos calculados con los precios de mercado del último período.")
        
        # Selector de rango temporal
        time_period = st.selectbox(
            "Seleccioná el período",
            ["1 mes", "3 meses", "6 meses", "1 año", "2 años"],
            index=3
        )
        
        period_map = {
            "1 mes": 30,
            "3 meses": 90,
            "6 meses": 180,
            "1 año": 365,
            "2 años": 730
        }
        
        days = period_map[time_period]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ticker, valuation_data in company_valuations.items():
            try:
                # Mapear ticker a formato yfinance
                yf_ticker = get_yfinance_ticker(ticker)
                
                # Datos históricos
                stock = yf.Ticker(yf_ticker)
                hist_data = stock.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    # Gráfico de comparación
                    fig_hist = go.Figure()
                    
                    # Precio de mercado histórico
                    fig_hist.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        name='Precio de mercado',
                        line=dict(color='blue', width=2),
                        mode='lines'
                    ))
                    
                    # Valor intrínseco calculado (línea horizontal)
                    intrinsic_value = valuation_data['stock_value']
                    fig_hist.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=[intrinsic_value] * len(hist_data),
                        name='Valor intrínseco calculado',
                        line=dict(color='red', width=2, dash='dash'),
                        mode='lines'
                    ))
                    
                    # Precio actual y diferencia
                    current_price = hist_data['Close'].iloc[-1]
                    difference_pct = ((intrinsic_value - current_price) / current_price) * 100
                    
                    fig_hist.update_layout(
                        title=f"{ticker} - Precio de mercado vs valor intrínseco",
                        xaxis_title="Fecha",
                        yaxis_title="Precio ($)",
                        height=400,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Métricas
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("Precio de mercado actual", f"${current_price:.2f}")
                    
                    with col_m2:
                        st.metric("Valor intrínseco calculado", f"${intrinsic_value:.2f}")
                    
                    with col_m3:
                        st.metric(
                            "Diferencia", 
                            f"{difference_pct:.2f}%",
                            delta=f"${intrinsic_value - current_price:.2f}",
                            delta_color="normal" if difference_pct > 0 else "inverse"
                        )
                    
                    if difference_pct > 0:
                        st.success(f"📊 {ticker} parecería **infravalorada** en {abs(difference_pct):.2f}% según el DCF")
                    elif difference_pct < 0:
                        st.warning(f"📊 {ticker} parecería **sobrevalorada** en {abs(difference_pct):.2f}% según el DCF")
                    else:
                        st.info(f"📊 {ticker} cotiza aproximadamente en su valor intrínseco calculado")
                    
                    st.markdown("---")
                else:
                    st.warning(f"No hay datos históricos disponibles para {ticker}")
                    
            except Exception as e:
                st.error(f"Error al obtener datos históricos para {ticker}: {str(e)}")
        
        st.divider()
    
    # Análisis de sensibilidad
    st.markdown(
    "<h3 style='color:#f89100ff; font-size:36px;'>5. Análisis de sensibilidad</h3>",
    unsafe_allow_html=True)

    # Parámetros de sensibilidad
    st.subheader("Parámetros de la matriz de sensibilidad")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    absolute_changes = st.checkbox('Cambio absoluto')
    
    with col_s1:
        wacc_sensitivity_pct = st.slider(
            "Rango de variación del WACC (±%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Porcentaje para variar el WACC alrededor del valor base"
        )
        wacc_sensitivity_range = wacc_sensitivity_pct / 100
        st.caption(f'Cambio de {wacc_sensitivity_pct*100} BPS')
        
    with col_s2:
        g_sensitivity_pct = st.slider(
            "Rango de variación de g (±%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Porcentaje para variar g alrededor del valor base"
        )
        g_sensitivity_range = g_sensitivity_pct / 100
        st.caption(f'Cambio de {g_sensitivity_pct*100} BPS')
        
    with col_s3:
        n_steps = st.slider("Número de cambios",
                            min_value=1,
                            max_value=30,
                            value=4,
                            step=1,
                            help="Cantidad de pasos desde el valor central de WACC/g")
    
    # Matrices por empresa
    st.subheader("Análisis de sensibilidad por empresa")
    tipo_de_cambio = 'absoluto' if absolute_changes else 'porcentual'
    st.markdown(f"*Los valores muestran el cambio {tipo_de_cambio} respecto del caso base*")
    
    # Toggle visualización
    viz_type = st.radio(
        "Tipo de visualización:",
        ["Tabla", "Mapa de calor", "Ambos"],
        horizontal=True,
        help="Elegí cómo mostrar los resultados del análisis de sensibilidad"
    )
    
    boton_precios = st.toggle('Cambios/Precios')
    
    all_sensitivity_matrices = []
    all_elasticidades = []
    
    for ticker, valuation_data in company_valuations.items():
        
        with st.expander(f"#### {ticker}"):
            
            # Arrays de WACC y g
            base_wacc = valuation_data['wacc_used']
            base_g = valuation_data['g_used']
            
            # Paso/cambio
            if absolute_changes:
                wacc_values = calcular_waccs2(base_wacc, cambio=wacc_sensitivity_range, n=n_steps)
                g_values = calcular_gs2(base_g, cambio=g_sensitivity_range, n=n_steps)
            else:
                wacc_values = calcular_waccs(base_wacc, cambio=wacc_sensitivity_range, n=n_steps)
                g_values = calcular_gs(base_g, cambio=g_sensitivity_range, n=n_steps)        
            
            # Matriz de sensibilidad
            sensitivity_matrix = dcf_sensitivity_matrix(
                wacc_values=wacc_values,
                g_values=g_values,
                free_cash_flows=valuation_data['fcf'],
                ticker=ticker,
                netdebt=valuation_data['netdebt'],
                g_to_use=g_rate
            )

            prices_matrix = dcf_scenarios(
                wacc_values=wacc_values,
                g_values=g_values,
                free_cash_flows=valuation_data['fcf'],
                ticker=ticker,
                netdebt=valuation_data['netdebt'],
                g_to_use=g_rate
            )
            
            elasticidades_df, valores_wacc, valores_g = crear_df_con_elasticidades(sensitivity_matrix)
        
            all_sensitivity_matrices.append(sensitivity_matrix)
            all_elasticidades.append(elasticidades_df)
            
            # Mostrar según selección
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
            
            if viz_type in ["Mapa de calor", "Ambos"]:
                # Heatmap interactivo
                fig = go.Figure(data=go.Heatmap(
                    z=sensitivity_matrix.values,
                    x=sensitivity_matrix.columns,
                    y=sensitivity_matrix.index,
                    colorscale='RdYlGn',
                    text=sensitivity_matrix.values,
                    texttemplate='%{text:.2f}%',
                    textfont={"size": 10},
                    colorbar=dict(title="% Cambio"),
                    hoverongaps=False,
                    hovertemplate='WACC: %{x}<br>g: %{y}<br>Cambio: %{z:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{ticker} - Mapa de calor de sensibilidad",
                    xaxis_title="WACC",
                    yaxis_title="Tasa de crecimiento (g)",
                    height=500,
                    font=dict(size=11)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    # Análisis agregado
    
    if len(all_sensitivity_matrices) > 1 and absolute_changes:
        columnas = valores_wacc
        indice = valores_g
        
        average_matrix = pd.DataFrame(sum([df.to_numpy() for df in all_sensitivity_matrices]) / len(all_sensitivity_matrices), columns=np.round(columnas), index=indice)
        
        st.subheader("Análisis agregado de sensibilidad (promedio de todas las empresas)")
        st.markdown("*Cambio porcentual promedio entre todas las empresas cargadas*")
        
        st.dataframe(average_matrix)
        
        st.markdown("*Desvío estándar*")
        desvio_matrix = pd.DataFrame(calcular_desvios(all_sensitivity_matrices), index=average_matrix.index, columns=average_matrix.columns)
        st.dataframe(desvio_matrix)
        
    if len(all_sensitivity_matrices) > 1:  
        elasti_prom = pd.DataFrame(sum([df.to_numpy() for df in all_elasticidades]) / len(all_elasticidades), columns=["Cambio relativo WACC (%)", "Elasticidades WACC promedio", "Cambio relativo g (%)", "Elasticidades g promedio"])
        desvio_elast = pd.DataFrame(calcular_desvios(all_elasticidades), columns=elasti_prom.columns)
        
        st.dataframe(elasti_prom)
        
    # Exportar resultados
    st.markdown(
    "<h3 style='color:#f89100ff; font-size:36px;'>6. Exportar resultados</h3>",
    unsafe_allow_html=True)
    
    def create_excel_report(company_valuations, all_sensitivity_matrices, avg_matrix=None):
        """Crear archivo de Excel con todas las valuaciones y matrices de sensibilidad"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Hoja resumen con valuaciones
            summary_data = []
            for ticker, valuation_data in company_valuations.items():
                summary_data.append({
                    'Ticker': ticker,
                    'Valor de la acción ($)': valuation_data['stock_value'],
                    'WACC usado': valuation_data['wacc_used'],
                    'Tasa de crecimiento (g)': valuation_data['g_used'],
                    'Años usados': valuation_data['years_used'],
                    'VT %': valuation_data['valor_terminal'], 
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Resumen de valuación', index=False)
            
            sheet_name = 'Sensibilidades'
            start_row = 0
            
            # Matrices individuales
            for idx, (ticker, valuation_data) in enumerate(company_valuations.items()):
                sensitivity_matrix = all_sensitivity_matrices[idx]
                elasticidad_matrix = all_elasticidades[idx]
                
                title_df = pd.DataFrame([[ticker]])
                title_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=False)
                title_df.to_excel(writer, sheet_name='Elasticidades', startrow=start_row, index=False, header=False)
                
                sensitivity_matrix.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1)
                elasticidad_matrix.to_excel(writer, sheet_name='Elasticidades', startrow=start_row + 1)
                
                start_row += n_steps * 2 + 4
            
            if absolute_changes and len(all_sensitivity_matrices) > 1:
                average_matrix.to_excel(writer, sheet_name='Promedios', startrow=1)
                desvio_matrix.to_excel(writer, sheet_name='Desvío', startrow=1)
                
            if len(all_sensitivity_matrices) > 1:
                elasti_prom.to_excel(writer, sheet_name='Elasticidades prom', startrow=1)
                desvio_elast.to_excel(writer, sheet_name='Elasticidades prom', startrow=len(elasti_prom.iloc[:,0])+3)
        
        output.seek(0)
        return output
    
    # Generar y ofrecer descarga (rejilla común para promedio)
    avg_matrix_for_download = None
    if len(all_sensitivity_matrices) > 1:
        # Recalcular con grilla común
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
    
    excel_report = create_excel_report(
        company_valuations, 
        all_sensitivity_matrices,
        avg_matrix_for_download
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.download_button(
        label="📥 Descargar informe en Excel",
        data=excel_report,
        file_name=f"DCF_Informe_Analisis_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Descargá un informe completo en Excel con valuaciones y matrices de sensibilidad"
    )
    
else:
    st.info("👆 Cargá uno o más archivos de Excel para comenzar el análisis")

# Pie
st.divider()
st.caption("Herramienta de valuación DCF y análisis de sensibilidad | Tickers soportados: " + ", ".join(acciones.keys()))

