# Análisis de Sensibilidad del Modelo DCF: WACC, g y Valor Intrínseco  
Repositorio asociado al Trabajo de Investigación Final (TIF) — Licenciatura en Finanzas Universidad Argentina de la Empresa (UADE)

Este repositorio contiene el código desarrollado para el Trabajo de Investigación Final (TIF), cuyo objetivo es analizar la sensibilidad del valor intrínseco calculado mediante el modelo de Flujos de Caja Descontados (Discounted Cash Flow, DCF) frente a variaciones controladas en el Costo Promedio Ponderado de Capital (WACC) y la tasa de crecimiento a perpetuidad (g).

El trabajo completo puede consultarse en el archivo PDF incluido en este repositorio.

---

## Objetivo del Trabajo

El estudio evalúa:
- El impacto de variaciones absolutas de hasta ±200 puntos básicos en WACC y g sobre el valor intrínseco.
- La elasticidad local del valor intrínseco ante variaciones relativas de hasta ±10% en ambos parámetros.
- El efecto marginal de WACC y g mediante una regresión log–log aplicada a 10 empresas del S&P 500.

Resultados principales:
- El valor intrínseco es altamente sensible al WACC, con una elasticidad promedio de –1.943.
- La sensibilidad a g es positiva y moderada, con una elasticidad promedio de 0.3536.
- El modelo presenta no linealidades y asimetrías, especialmente ante reducciones del WACC o aumentos de g.

---

## Metodología (Resumen)

El análisis utiliza:
- Valuación por DCF bajo el enfoque de FCFF.
- Proyecciones de Bloomberg (EE – Bloomberg Estimates) para FCFF, WACC y supuestos financieros (2025–2030).
- Tasa de crecimiento a perpetuidad (g) igual a 3%.
- Simulaciones de sensibilidad univariada y bivariada.
- Estimación econométrica mediante una regresión log–log:

ln(Valor) = α + β1 ln(WACC) + β2 ln(g) + ε

Empresas analizadas (Top 10 S&P 500 a agosto 2025): AAPL, AMZN, AVGO, GOOGL, META, MSFT, NVDA, ORCL, TSLA, WMT.

---

## Herramientas Utilizadas

- Python 3.10+
- Pandas, NumPy
- Plotly
- Statsmodels
- Streamlit
- yfinance / Bloomberg Estimates

---

## Cómo Ejecutar la Aplicación

1. Clonar el repositorio:
git clone https://github.com/usuario/dcf-sensitivity-analysis.git
cd dcf-sensitivity-analysis

2. Crear y activar un entorno virtual:
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

3. Instalar dependencias:
pip install -r requirements.txt

4. Ejecutar la aplicación:
streamlit run app.py

---

## Funcionalidades Principales

- Cálculo del valor intrínseco mediante FCFF.
- Matrices de sensibilidad (WACC × g).
- Heatmaps interactivos.
- Cálculo de elasticidades locales.
- Comparación entre empresas y valor promedio.
- Modelo econométrico replicable.
- Exportación de tablas y gráficos.

---

## Reproducibilidad

El proyecto es completamente replicable gracias a:
- Código modular en funciones.py.
- Scripts de sensibilidad en app.py.
- Notebook reproducible para la regresión log–log.
- Datos base obtenidos de Bloomberg.
- Visualizaciones generadas automáticamente.

---

## Referencias Principales

- Damodaran, A. (2012, 2011, 2025). Investment Valuation.
- Koller, T. et al. (2005). Valuation: Measuring and Managing the Value of Companies.
- Saltelli, A. et al. (2004). Sensitivity Analysis.
- Pinto, H. et al. (2010). Equity Asset Valuation.
- Sampieri, R. et al. (2014). Metodología de la Investigación.

La bibliografía completa se encuentra en el PDF del TIF.

---

## Autores
Nicolas Etienne Prandi\nAdrian Félix Parisi

Trabajo realizado en el marco del Trabajo de Investigación Final (TIF) de la Licenciatura en Finanzas de la Universidad Argentina de la Empresa (UADE). Incluye análisis empírico, código en Python y visualizaciones desarrolladas con Streamlit.

---

## Licencia

Proyecto destinado exclusivamente para fines académicos, educativos y no comerciales.
