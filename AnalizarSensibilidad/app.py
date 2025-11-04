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

# py -m streamlit run app.py --server.address localhost --server.port 8501

# Ticker mapping for yfinance (some tickers need special formatting)
def get_yfinance_ticker(ticker):
    """Map internal tickers to yfinance-compatible format"""
    ticker_map = {
        'BRK': 'BRK-B',  # Berkshire Hathaway Class B
    }
    return ticker_map.get(ticker, ticker)

# Configure page
st.set_page_config(
    layout="wide"
)

st.markdown(
    "<h1 style='color:#f89100ff; font-size:42px;'>📊 DCF Valuation & Sensitivity Analysis</h1>",
    unsafe_allow_html=True
)

# Title and description
st.markdown("Upload Excel files containing financial data to perform DCF valuation and sensitivity analysis")

# Initialize session state for processed data
if 'processed_companies' not in st.session_state:
    st.session_state.processed_companies = {}

# Initialize session state for scenarios
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []

# File upload section
st.header("1. Upload Financial Data")
uploaded_files = st.file_uploader(
    "Upload one or more Excel files",
    type=['xlsx', 'xls'],
    accept_multiple_files=True,
    help="Upload Excel files containing financial statements for company analysis"
)

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        
        # Only process if not already processed
        if file_key not in st.session_state.processed_companies:
            try:
                # Read Excel file
                excel_data = pd.read_excel(uploaded_file)
                
                # Process the data through the pipeline
                excel_limpio, ticker = limpiar_excel(excel_data)
                
                # Verify ticker exists in our data
                if ticker not in acciones:
                    st.error(f"❌ Ticker '{ticker}' from file '{uploaded_file.name}' is not supported. Supported tickers: {', '.join(acciones.keys())}")
                    continue
                
                # Extract and process financial items
                partidas = extraer_partidas(excel_limpio)
                partidas.index = partidas.index.map(normalizar_indice)
                partidas = limpiar_partidas(partidas)
                
                # Calculate net debt
                netdebt = get_netdebt(partidas)
                
                # Complete and calculate new items
                partidas = completar_partidas(partidas)
                
                partidas = calcular_nuevas_partidas(partidas)
                
                # Get stable FCFF
                anio_corte, fcff_hasta_corte = elegir_ultimo_fcff_estable(
                    partidas,
                    fila_fcff="FCFF",
                    k_pos=4
                )
                
                free_cash_flows = np.array(fcff_hasta_corte)[1:]
                
                # Store processed data
                st.session_state.processed_companies[file_key] = {
                    'ticker': ticker,
                    'partidas': partidas,
                    'netdebt': netdebt,
                    'free_cash_flows': free_cash_flows,
                    'anio_corte': anio_corte,
                    'default_wacc': acciones[ticker].wacc,
                    'num_years': len(free_cash_flows)
                }
                
                st.success(f"✅ Successfully processed: {uploaded_file.name} (Ticker: {ticker})")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

# Display analysis if we have processed companies
if st.session_state.processed_companies:
    # Additional Information
    with st.expander("📋 View Processed Financial Data"):
        for file_key, company_data in st.session_state.processed_companies.items():
            st.markdown(f"### {company_data['ticker']} - {file_key}")
            st.dataframe(company_data['partidas'].loc[['NOPAT',"D&A",'CAPEX','WK','FCFF']], use_container_width=True)
            st.markdown("---")
    
    st.markdown(
        "<h3 style='color:#f89100ff; font-size:36px;'>2. Analysis Parameters</h1>",
        unsafe_allow_html=True)
    
    # Get min/max values for sliders based on all companies
    all_waccs = [data['default_wacc'] for data in st.session_state.processed_companies.values()]
    all_num_years = [data['num_years'] for data in st.session_state.processed_companies.values()]
    
    min_wacc = min(all_waccs)
    max_wacc = max(all_waccs)
    avg_wacc = np.mean(all_waccs)
    
    max_years = max(all_num_years)
    
    # Create three columns for parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        g_rate = st.slider(
            "Growth Rate (g)",
            min_value=0.0,
            max_value=0.10,
            value=0.03,
            step=0.001,
            format="%.3f",
            help="Terminal growth rate for perpetuity calculation"
        )
        st.caption(f"Selected: {g_rate:.1%}")
    
    with col2:
        # Get unique default WACCs and use average as default
        wacc_override = st.slider(
            "WACC Override",
            min_value=0.01,
            max_value=0.25,
            value=float(avg_wacc),
            step=0.001,
            format="%.3f",
            help="Weighted Average Cost of Capital (will use company default if not changed)"
        )
        st.caption(f"Selected: {wacc_override:.1%}")
    
    with col3:
        num_years_to_use = st.slider(
            "Number of Years for DCF",
            min_value=1,
            max_value=max_years,
            value=max_years,
            step=1,
            help="Number of years of cash flows to use in the valuation"
        )
        st.caption(f"Using {num_years_to_use} years")
    
    # Use company default WACC checkbox
    use_default_wacc = st.checkbox(
        "Use each company's default WACC",
        value=True,
        help="If checked, each company will use its own default WACC. Otherwise, the WACC Override value will be used for all companies."
    )
    
    # Scenario Analysis
    st.subheader("💾 Scenario Management")
    
    col_scenario1, col_scenario2 = st.columns([2, 1])
    
    with col_scenario1:
        scenario_name = st.text_input(
            "Scenario Name",
            placeholder="e.g., Base Case, Optimistic, Conservative",
            help="Give this scenario a descriptive name"
        )
    
    with col_scenario2:
        if st.button("💾 Save Current Scenario", type="primary", use_container_width=True):
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
                st.success(f"✅ Scenario '{scenario_name}' saved!")
            else:
                st.error("Please enter a scenario name")
    
    # Display saved scenarios
    if st.session_state.saved_scenarios:
        with st.expander(f"📊 View & Compare Saved Scenarios ({len(st.session_state.saved_scenarios)})", expanded=False):
            # Display scenarios table
            scenario_df = pd.DataFrame(st.session_state.saved_scenarios)
            scenario_display = scenario_df.copy()
            scenario_display['g_rate'] = scenario_display['g_rate'].apply(lambda x: f"{x:.2%}")
            scenario_display['wacc_override'] = scenario_display['wacc_override'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                scenario_display[['name', 'g_rate', 'wacc_override', 'num_years', 'use_default_wacc', 'timestamp']],
                use_container_width=True
            )
            
            # Scenario comparison feature
            st.subheader("Compare Scenarios")
            selected_scenarios = st.multiselect(
                "Select scenarios to compare",
                options=[s['name'] for s in st.session_state.saved_scenarios],
                help="Select 2 or more scenarios to compare their results"
            )
            
            if len(selected_scenarios) >= 2:
                # Calculate valuations for each selected scenario
                comparison_data = []
                
                for scenario_name in selected_scenarios:
                    scenario = next(s for s in st.session_state.saved_scenarios if s['name'] == scenario_name)
                    
                    # Calculate valuations for each company with this scenario's parameters
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
                            'Scenario': scenario_name,
                            'Ticker': ticker,
                            'Stock Value ($)': stock_value,
                            'Terminal Value': valor_terminal,
                            'g': f"{scenario['g_rate']:.2%}",
                            'WACC': f"{wacc_to_use:.2%}"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create comparison chart
                fig_comparison = px.bar(
                    comparison_df,
                    x='Ticker',
                    y='Stock Value ($)',
                    color='Scenario',
                    barmode='group',
                    title='Scenario Comparison - Stock Valuations',
                    text='Stock Value ($)'
                )
                
                fig_comparison.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                fig_comparison.update_layout(height=500)
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Show comparison table
                st.dataframe(comparison_df, use_container_width=True)
            
            col_clear, col_space = st.columns([1, 3])
            with col_clear:
                if st.button("🗑️ Clear All Scenarios"):
                    st.session_state.saved_scenarios = []
                    st.rerun()
    
    st.divider()
    
    # Valuation Results
    st.markdown(
    "<h3 style='color:#f89100ff; font-size:35px;'>3. DCF Valuation Results</h1>",
    unsafe_allow_html=True)

    # Store valuations for each company
    company_valuations = {}
    
    # Create columns for displaying results
    num_companies = len(st.session_state.processed_companies)
    cols = st.columns(min(3, num_companies))
    
    for idx, (file_key, company_data) in enumerate(st.session_state.processed_companies.items()):
        ticker = company_data['ticker']
        
        # Determine which WACC to use
        wacc_to_use = company_data['default_wacc'] if use_default_wacc else wacc_override
        
        # Slice free cash flows based on num_years_to_use
        fcf_to_use = company_data['free_cash_flows'][:num_years_to_use]
        
        # Calculate valuation
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
        
        # Display in column
        col_idx = idx % 3
        with cols[col_idx]:
            st.metric(
                label=f"{ticker}",
                value=f"${stock_value:.2f}",
                delta=None
            )
            st.caption(f"WACC: {wacc_to_use:.2%}")
            st.caption(f"Growth: {g_rate:.2%}")
            st.caption(f"Terminal Value (%): {valor_terminal:.2%}")
            st.caption(f"Years: {num_years_to_use}")
    
    st.divider()
    
    # Comparative Charts
    if len(company_valuations) > 1:
        st.markdown(
        "<h3 style='color:#f89100ff; font-size:36px;'>4. Comparative Analysis</h1>",
        unsafe_allow_html=True)
        
        # Prepare data for charts
        tickers_list = list(company_valuations.keys())
        stock_values = [val['stock_value'] for val in company_valuations.values()]
        waccs_used = [val['wacc_used'] for val in company_valuations.values()]
        gs_used = [val['g_used'] for val in company_valuations.values()]
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Stock value comparison
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
                        colorbar=dict(title="Value ($)")
                    )
                )
            ])
            
            fig_value.update_layout(
                title="Stock Value Comparison",
                xaxis_title="Company",
                yaxis_title="Stock Value ($)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_value, use_container_width=True)
        
        with col_chart2:
            # WACC comparison
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
                title="WACC Comparison",
                xaxis_title="Company",
                yaxis_title="WACC (%)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_wacc, use_container_width=True)
        
        # Combined metrics chart
        fig_combined = go.Figure()
        
        fig_combined.add_trace(go.Scatter(
            x=tickers_list,
            y=stock_values,
            name='Stock Value',
            mode='lines+markers',
            yaxis='y',
            line=dict(width=3),
            marker=dict(size=10)
        ))
        
        fig_combined.update_layout(
            title="Valuation Metrics Across Companies",
            xaxis_title="Company",
            yaxis_title="Stock Value ($)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
        st.divider()
    
    # Historical Price Comparison
    st.header("4.5. Historical Price vs Intrinsic Value")
    
    with st.expander("📈 View Historical Price Comparison", expanded=False):
        st.markdown("Compare calculated intrinsic values with actual market prices over the past year")
        
        # Time range selector
        time_period = st.selectbox(
            "Select Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years"],
            index=3
        )
        
        period_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730
        }
        
        days = period_map[time_period]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ticker, valuation_data in company_valuations.items():
            try:
                # Map ticker to yfinance format
                yf_ticker = get_yfinance_ticker(ticker)
                
                # Fetch historical data from yfinance
                stock = yf.Ticker(yf_ticker)
                hist_data = stock.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    # Create comparison chart
                    fig_hist = go.Figure()
                    
                    # Historical market price
                    fig_hist.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=hist_data['Close'],
                        name='Market Price',
                        line=dict(color='blue', width=2),
                        mode='lines'
                    ))
                    
                    # Calculated intrinsic value (horizontal line)
                    intrinsic_value = valuation_data['stock_value']
                    fig_hist.add_trace(go.Scatter(
                        x=hist_data.index,
                        y=[intrinsic_value] * len(hist_data),
                        name='Calculated Intrinsic Value',
                        line=dict(color='red', width=2, dash='dash'),
                        mode='lines'
                    ))
                    
                    # Calculate current market price and difference
                    current_price = hist_data['Close'].iloc[-1]
                    difference_pct = ((intrinsic_value - current_price) / current_price) * 100
                    
                    fig_hist.update_layout(
                        title=f"{ticker} - Market Price vs Intrinsic Value",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
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
                    
                    # Show comparison metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("Current Market Price", f"${current_price:.2f}")
                    
                    with col_m2:
                        st.metric("Calculated Intrinsic Value", f"${intrinsic_value:.2f}")
                    
                    with col_m3:
                        st.metric(
                            "Difference", 
                            f"{difference_pct:.2f}%",
                            delta=f"${intrinsic_value - current_price:.2f}",
                            delta_color="normal" if difference_pct > 0 else "inverse"
                        )
                    
                    if difference_pct > 0:
                        st.success(f"📊 {ticker} appears **undervalued** by {abs(difference_pct):.2f}% based on DCF analysis")
                    elif difference_pct < 0:
                        st.warning(f"📊 {ticker} appears **overvalued** by {abs(difference_pct):.2f}% based on DCF analysis")
                    else:
                        st.info(f"📊 {ticker} is trading at approximately its calculated intrinsic value")
                    
                    st.markdown("---")
                else:
                    st.warning(f"No historical data available for {ticker}")
                    
            except Exception as e:
                st.error(f"Error fetching historical data for {ticker}: {str(e)}")
        
        st.divider()
    
    # Sensitivity Analysis
    st.markdown(
    "<h3 style='color:#f89100ff; font-size:36px;'>5. Sensitivity Analysis</h1>",
    unsafe_allow_html=True)


    # Parameters for sensitivity analysis
    st.subheader("Sensitivity Matrix Parameters")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    absolute_changes = st.checkbox('Absolute Change')
    
    with col_s1:
        wacc_sensitivity_pct = st.slider(
            "WACC Variation Range (±%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Percentage range to vary WACC around the base value"
        )
        wacc_sensitivity_range = wacc_sensitivity_pct / 100
        st.caption(f'Cambio de {wacc_sensitivity_pct*100} BPS')
        
    with col_s2:
        g_sensitivity_pct = st.slider(
            "Growth Rate Variation Range (±%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Percentage range to vary growth rate around the base value"
        )
        g_sensitivity_range = g_sensitivity_pct / 100
        st.caption(f'Cambio de {g_sensitivity_pct*100} BPS')
        
    with col_s3:
        n_steps = st.slider("Number of changes",
                            min_value=1,
                            max_value=10,
                            value=4,
                            step=1,
                            help="Number of steps from the central value of WACC/g")
    
    # Generate sensitivity matrices for each company
    st.subheader("Individual Company Sensitivity Analysis")
    tipo_de_cambio = 'absolute' if absolute_changes else 'percentage'
    st.markdown(f"*Values show {tipo_de_cambio} change from base case valuation*")
    
    # Add visualization toggle
    viz_type = st.radio(
        "Visualization Type:",
        ["Table", "Heatmap", "Both"],
        horizontal=True,
        help="Choose how to display sensitivity analysis results"
    )

    boton_precios = st.toggle('Changes/Prices')
    
    all_sensitivity_matrices = []
    
    all_elasticidades = []
    
    for ticker, valuation_data in company_valuations.items():
        
        with st.expander(f"#### {ticker}"):
        
            # st.markdown(f"#### {ticker}")
            
            # Calculate WACC and g arrays for sensitivity
            base_wacc = valuation_data['wacc_used']
            base_g = valuation_data['g_used']
            
            # cambio parameter represents the step size, n=5 gives 11 points (5 below, center, 5 above)
            # To get the desired range, divide by n
            if absolute_changes:
                wacc_values = calcular_waccs2(base_wacc, cambio=wacc_sensitivity_range, n=n_steps)
                g_values = calcular_gs2(base_g, cambio=g_sensitivity_range, n=n_steps)
                
            else:
                wacc_values = calcular_waccs(base_wacc, cambio=wacc_sensitivity_range, n=n_steps)
                g_values = calcular_gs(base_g, cambio=g_sensitivity_range, n=n_steps)        
            
            # Generate sensitivity matrix
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
            
            elasticidades_df = crear_df_con_elasticidades(sensitivity_matrix)[0]
            
            valores_wacc = crear_df_con_elasticidades(sensitivity_matrix)[1]
            
            valores_g = crear_df_con_elasticidades(sensitivity_matrix)[2]
        
            all_sensitivity_matrices.append(sensitivity_matrix)
            
            all_elasticidades.append(elasticidades_df)
            
            # Display based on selected visualization type
            if viz_type in ["Table", "Both"]:
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
            
            if viz_type in ["Heatmap", "Both"]:
                # Create interactive Plotly heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=sensitivity_matrix.values,
                    x=sensitivity_matrix.columns,
                    y=sensitivity_matrix.index,
                    colorscale='RdYlGn',
                    text=sensitivity_matrix.values,
                    texttemplate='%{text:.2f}%',
                    textfont={"size": 10},
                    colorbar=dict(title="% Change"),
                    hoverongaps=False,
                    hovertemplate='WACC: %{x}<br>g: %{y}<br>Change: %{z:.2f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"{ticker} Sensitivity Heatmap",
                    xaxis_title="WACC",
                    yaxis_title="Growth Rate (g)",
                    height=500,
                    font=dict(size=11)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    # Aggregate sensitivity analysis
    
    if len(all_sensitivity_matrices) > 1 and absolute_changes:
        columnas = valores_wacc
        indice = valores_g
        
        average_matrix = pd.DataFrame(sum([df.to_numpy() for df in all_sensitivity_matrices]) / len(all_sensitivity_matrices), columns=np.round(columnas), index=indice)
        
        st.subheader("Aggregate Sensitivity Analysis (Average Across All Companies)")
        st.markdown("*Average percentage change across all uploaded companies*")
        
        st.dataframe(average_matrix)
        
        st.markdown("*Standard Deviation*")
        desvio_matrix = pd.DataFrame(calcular_desvios(all_sensitivity_matrices), index=average_matrix.index, columns=average_matrix.columns)
        st.dataframe(desvio_matrix)

    if len(all_sensitivity_matrices) > 1:  
        elasti_prom = pd.DataFrame(sum([df.to_numpy() for df in all_elasticidades]) / len(all_elasticidades), columns=["Cambio Relativo WACC (%)", "Elasticidades WACC Promedio", "Cambio Relativo g (%)", "Elasticidades g Promedio"])
        desvio_elast = pd.DataFrame(calcular_desvios(all_elasticidades), columns=elasti_prom.columns)
        
        st.dataframe(elasti_prom)
    
    # Download Excel Report
    st.markdown(
    "<h3 style='color:#f89100ff; font-size:36px;'>6. Export Results</h1>",
    unsafe_allow_html=True)
    
    def create_excel_report(company_valuations, all_sensitivity_matrices, avg_matrix=None):
        """Create Excel file with all valuation results and sensitivity matrices"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet with all valuations
            summary_data = []
            for ticker, valuation_data in company_valuations.items():
                summary_data.append({
                    'Ticker': ticker,
                    'Stock Value ($)': valuation_data['stock_value'],
                    'WACC Used': valuation_data['wacc_used'],
                    'Growth Rate (g)': valuation_data['g_used'],
                    'Years Used': valuation_data['years_used'],
                    'TV %': valuation_data['valor_terminal'], 
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Valuation Summary', index=False)
            
            sheet_name = 'All sensitivities'
            start_row = 0
            
            # Individual sensitivity matrices
            for idx, (ticker, valuation_data) in enumerate(company_valuations.items()):
                sensitivity_matrix = all_sensitivity_matrices[idx]
                elasticidad_matrx = all_elasticidades[idx]
                
                title_df = pd.DataFrame([[ticker]])
                title_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False, header=False)
                title_df.to_excel(writer, sheet_name='Elasticidades', startrow=start_row, index=False, header=False)
                
                sensitivity_matrix.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1)
                elasticidad_matrx.to_excel(writer, sheet_name='Elasticidades', startrow=start_row + 1)
                
                start_row += n_steps * 2 + 4
            
            if absolute_changes and len(all_sensitivity_matrices) > 1:
                average_matrix.to_excel(writer, sheet_name='Averages', startrow=1)
                desvio_matrix.to_excel(writer, sheet_name='Std', startrow=1)

            if len(all_sensitivity_matrices) > 1:
                elasti_prom.to_excel(writer, sheet_name='Prom Elast', startrow=1)
                desvio_elast.to_excel(writer, sheet_name='Prom Elast', startrow=len(elasti_prom.iloc[:,0])+3)
        
        output.seek(0)
        return output
    
    # Generate and offer download (use common grid for average matrix)
    avg_matrix_for_download = None
    if len(all_sensitivity_matrices) > 1:
        # Recalculate with common grid for consistency
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
        label="📥 Download Excel Report",
        data=excel_report,
        file_name=f"DCF_Analysis_Report_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download a comprehensive Excel report with all valuation results and sensitivity matrices"
    )
    
else:
    st.info("👆 Please upload one or more Excel files to begin the analysis")

# Footer
st.divider()
st.caption("DCF Valuation & Sensitivity Analysis Tool | Supported tickers: " + ", ".join(acciones.keys()))
