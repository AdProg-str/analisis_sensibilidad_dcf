# DCF Valuation & Sensitivity Analysis Application

## Overview

This is a comprehensive financial analysis application built with Streamlit that performs Discounted Cash Flow (DCF) valuation and sensitivity analysis on major US companies. The application allows users to upload Excel files containing financial statements and generates detailed valuation models with interactive visualizations, comparative analytics, and scenario planning capabilities.

The system focuses on 10 major companies (AAPL, AMZN, AVGO, BRK, GOOGL, JPM, META, MSFT, NVDA, TSLA) with pre-configured WACC, equity ratios, and share counts. It processes financial data through a pipeline that cleans, normalizes, calculates free cash flows, and performs DCF valuations with sensitivity matrices.

## Recent Changes

**Date**: October 2025
- Added Excel export functionality for valuation results and sensitivity matrices
- Implemented interactive Plotly heatmaps for sensitivity analysis with table/heatmap toggle
- Added comparative charts for multi-company analysis (bar charts, line charts)
- Integrated historical stock price comparison using yfinance API
- Implemented scenario management system for saving and comparing different parameter sets
- Fixed sensitivity range calculation to provide accurate variation ranges
- Improved average sensitivity matrix calculation using common WACC/g grids
- Added ticker mapping for yfinance API compatibility (e.g., BRK → BRK-B)

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit web application
- **Decision**: Use Streamlit for rapid development of data-driven financial applications
- **Rationale**: Streamlit provides native support for data visualization, file uploads, and interactive widgets without requiring frontend framework expertise
- **Components**:
  - File upload interface for multiple Excel files
  - Session state management for processed companies and saved scenarios
  - Wide layout configuration for displaying complex financial data and charts

### Data Processing Pipeline

**ETL Architecture**: Sequential function-based pipeline
- **Decision**: Implement data processing as discrete, chainable functions rather than OOP classes
- **Rationale**: Financial calculations follow a linear workflow; functional approach provides clarity and testability
- **Pipeline Stages**:
  1. `limpiar_excel`: Clean and normalize Excel input (remove headers, standardize column names, extract years)
  2. `extraer_partidas`: Extract specific financial line items from statements
  3. `normalizar_indice`: Normalize index values for consistent data structure
  4. `limpiar_partidas`: Clean extracted line items (handle missing data, format numbers)
  5. `get_netdebt`: Calculate net debt position
  6. `completar_partidas`: Fill in missing financial data points
  7. `calcular_nuevas_partidas`: Derive calculated metrics (margins, ratios, growth rates)
  8. `elegir_ultimo_fcff_estable`: Select stable terminal free cash flow
  9. `valuacion_DCF`: Perform DCF valuation calculation
  10. `calcular_waccs` & `calcular_gs`: Generate WACC and growth rate ranges
  11. `dcf_sensitivity_matrix`: Create sensitivity analysis grid

**Data Model**: Company-specific objects (Accion class)
- **Decision**: Pre-configure valuation parameters for each stock ticker
- **Rationale**: Major companies have relatively stable capital structures; hardcoding reduces user input requirements
- **Attributes**: WACC (weighted average cost of capital), equity ratio, outstanding shares

### Visualization Layer

**Library**: Plotly (graph_objects and express modules)
- **Decision**: Use Plotly for interactive financial charts
- **Rationale**: Plotly provides interactive features (zoom, hover, pan) essential for analyzing sensitivity matrices and time-series financial data
- **Alternatives Considered**: Matplotlib (less interactive), Altair (less feature-rich)

### State Management

**Approach**: Streamlit session state
- **Decision**: Use `st.session_state` for persisting processed data and saved scenarios
- **Rationale**: Prevents re-processing uploaded files on widget interactions; enables scenario comparison
- **State Objects**:
  - `processed_companies`: Dictionary mapping tickers to processed financial data
  - `saved_scenarios`: List of user-defined valuation scenarios with different assumptions

## External Dependencies

### Third-Party Libraries

**yfinance**: Yahoo Finance API wrapper
- **Purpose**: Fetch real-time stock prices for comparison with intrinsic valuations
- **Usage**: Ticker mapping function (`get_yfinance_ticker`) handles special cases (e.g., BRK → BRK-B)

**pandas**: Data manipulation and analysis
- **Purpose**: Core data structure for financial statements and calculations
- **Usage**: DataFrame operations for ETL pipeline, time-series analysis

**numpy**: Numerical computing
- **Purpose**: Array operations and mathematical calculations in valuation formulas
- **Usage**: DCF calculations, sensitivity matrix generation

**streamlit**: Web application framework
- **Purpose**: User interface and application hosting
- **Configuration**: Wide layout, custom page title and icon

**plotly**: Interactive visualization
- **Purpose**: Generate interactive charts for financial analysis
- **Modules**: `graph_objects` for custom charts, `express` for rapid prototyping

### Data Sources

**Input Format**: Excel files (.xlsx, .xls)
- **Structure**: Financial statements with specific row/column format
- **Expected Content**: Income statement, balance sheet, cash flow statement line items
- **Processing**: Custom parsing logic extracts ticker from first cell, removes header rows, standardizes year columns

**Company Data**: Hardcoded in `acciones` dictionary
- **Companies**: 10 major US stocks (AAPL, AMZN, AVGO, BRK, GOOGL, JPM, META, MSFT, NVDA, TSLA)
- **Parameters**: WACC, equity percentage, shares outstanding
- **Note**: Values appear to be static snapshots; no dynamic update mechanism from external APIs

### File I/O

**Excel Processing**: Pandas `read_excel` (implied)
- **Purpose**: Load financial data from uploaded Excel files
- **Format**: Multi-sheet Excel workbooks with standardized financial statement structure

**In-Memory Storage**: io.BytesIO (imported but usage not shown in provided code)
- **Purpose**: Likely used for downloading processed results or scenario exports