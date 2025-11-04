import pandas as pd
import numpy as np
import yfinance as yf
import re
import math

class Accion():
    def __init__(self, wacc, shares):
        self.wacc = wacc
        self.shares = shares  

acciones = {
    "AAPL": Accion(0.102, 15408),
    "AMZN": Accion(0.109, 10721),
    "AVGO": Accion(0.149, 4778),
    "GOOGL": Accion(0.101, 12447),
    "META": Accion(0.109, 2614),
    "MSFT": Accion(0.097, 7469),
    "NVDA": Accion(0.151, 24940),
    "TSLA": Accion(0.132, 3498),
    "ORCL": Accion(0.109, 2866),
    "WMT": Accion(0.083, 8081),
}

def limpiar_excel(excel):
    excel.index = excel.iloc[:,0]
    ticker = excel.index[0].split()[0]
    excel_limpio = excel.drop(excel.columns[0], axis=1)
    excel_limpio = excel_limpio.drop(excel_limpio.index[0])
    excel_limpio.columns = excel_limpio.iloc[0]
    excel_limpio = excel_limpio.drop(excel_limpio.index[0])
    excel_limpio = excel_limpio.drop(excel_limpio.iloc[:,0:3], axis=1)
    
    anios = [anio[:4] for anio in list(excel_limpio.columns)]

    excel_limpio.columns = anios
    
    return pd.DataFrame(excel_limpio), ticker

def extraer_partidas(excel_limpio):
    patterns = [
        r'^(?!.*\b(?:Re)?Payments\b).*(?:Total Debt$|Debt$|Net(?=.*\bdebt\b))',
        r'Equivalents\s\w*\sShort|\bEquivalents\b.*\bMarketable\b|\bCash\b.*Marketable|\bCash\b.*Short-term|\bCash\b.*Equivalents$',
        r'^\s*?Tax Rate\s\(\%\)',
        r'^\s*?Revenue$|^\s*?Net Revenue',
        r'^\s*Operating Margin\s\(\%\)',
        r'(?:Depreciation\s*(?:&|and)\s*Amortization|D\s*&\s*A)',
        r'(?:Capital\s*Expenditures?$|CAPEX|Capex)',
        r'^\s*Current Assets$',
        r'^\s*Current Liabilities$'
    ]

    pat = re.compile('|'.join(patterns), flags=re.IGNORECASE)

    mask = excel_limpio.index.astype(str).str.contains(pat, regex=True, na=False)
    partidas = excel_limpio.loc[mask]

    partidas.index = partidas.index.str.strip()
    
    partidas = partidas.astype(float)
    
    return partidas

def normalizar_indice(nombre):
    nombre = nombre.strip().lower()

    patrones = {
        r"total(?=.*\bdebt\b)": "Total Debt",
        r"short(?=.*\bdebt\b)": "ST Debt",
        r"net(?=.*\bdebt\b)": "Net Debt",
        r"debt": "LT Debt",
        r"cash|marketable": "Cash",
        r"revenue": "Ventas",
        r"tax": 'Tax',
        r"margin": "Opmg",
        r"nopat": "NOPAT",
        r"depreciation.*amortization": "D&A",
        r"capex|capital expenditures?": "CAPEX",
        r"current assets": "Current Assets",
        r"current liabilities": "Current Liabilities"
    }

    for patron, nuevo in patrones.items():
        if re.search(patron, nombre, re.IGNORECASE):
            return nuevo
    
    return nombre  


def limpiar_partidas(partidas):
    partidas = partidas[~partidas.isnull().all(axis=1)] 
    
    partidas = partidas.fillna(0)

    partidas = partidas.drop_duplicates()

    cuentas = partidas.index

    for cuenta in cuentas:
        if type(partidas.loc[cuenta]) == pd.core.frame.DataFrame:

            operando = partidas.loc[cuenta]
            fila_max = operando[operando.sum(axis=1) == operando.sum(axis=1).max()]

            partidas = partidas.drop(cuenta)
            
            partidas.loc[cuenta] = fila_max.loc[cuenta]
            
    return partidas

def get_netdebt(partidas):
    if 'Net Debt' in partidas.index:
        netdebt = float(partidas.loc['Net Debt', partidas.columns[0]])
        return netdebt
    
    elif 'Total Debt' in partidas.index:
        netdebt = float(partidas.loc['Total Debt', partidas.columns[0]] - partidas.loc['Cash', partidas.columns[0]])
        return netdebt 
    
    elif 'ST Debt' in partidas.index and 'LT Debt' in partidas.index:
        netdebt = float(partidas.loc['ST Debt', partidas.columns[0]] + partidas.loc['LT Debt', partidas.columns[0]] - partidas.loc['Cash', partidas.columns[0]]) 
        return netdebt
    
    else:
        netdebt = float(partidas.loc['LT Debt', partidas.columns[0]] - partidas.loc['Cash', partidas.columns[0]]) 
        return netdebt
    
def completar_partidas(partidas):
    for cuenta in partidas.index:
        if partidas.loc[cuenta].eq(0).any():

            porcentaje = partidas.loc[cuenta] / partidas.loc['Ventas']

            porcentaje_medio = porcentaje[porcentaje != 0].mean()

            resultados = partidas.loc[cuenta][partidas.loc[cuenta] == 0]

            ventas = partidas.loc['Ventas', resultados.index]

            partidas.loc[cuenta, partidas.loc[cuenta] == 0] = ventas * porcentaje_medio
            
    return partidas

def calcular_nuevas_partidas(partidas):
    partidas.loc['WK'] = (partidas.loc['Current Assets'] - partidas.loc['Cash']) - partidas.loc['Current Liabilities']

    partidas.loc['Change WK'] = partidas.loc['WK'].diff()

    partidas.loc['NOPAT'] = partidas.loc['Ventas'] * (partidas.loc['Opmg'] / 100) * (1-partidas.loc['Tax']/100)

    partidas.loc['FCFF'] = partidas.loc['NOPAT'] + partidas.loc['D&A'] + partidas.loc['CAPEX'] - partidas.loc['Change WK']

    return partidas

def elegir_ultimo_fcff_estable(df,
                               fila_fcff="FCFF",
                               k_pos=4,                           
                               ):
    """
    Devuelve:
      - anio_corte (columna)
      - serie_fcff_cortada (hasta anio_corte inclusive)
    Asume df con años en columnas y una fila 'FCFF'.
    """
    fcff = df.loc[fila_fcff].astype(float)  

    positivos = (fcff > 0).astype(int)
    
    racha_pos = positivos.rolling(k_pos, min_periods=k_pos).sum() == k_pos

    mascara = (fcff > 0) & racha_pos 

    candidatos = mascara[mascara].index
    if len(candidatos) > 0:
        anio_corte = candidatos[-1]
    else:
        pos = (fcff > 0)
        if pos.any():
            anio_corte = pos[pos].index[-1]
        else:
            raise ValueError("No hay ningún FCFF positivo; revisá supuestos.")

    serie_fcff_cortada = fcff.loc[:anio_corte]
    return anio_corte, serie_fcff_cortada

def valuacion_DCF(wacc, g, free_cash_flows, ticker, netdebt):
    
    shares = acciones[ticker].shares
    
    ddcf = []

    for año, cashflow in enumerate(free_cash_flows):
        if año != len(free_cash_flows)-1:
            descuento = cashflow / (1+wacc)**(año+1)
            ddcf.append(descuento)
        else:
            descuento = cashflow / (1+wacc)**(año+1)
            perpetuidad = (cashflow * (1+g)) / (((1+wacc)**(año+1)) * (wacc - g))
            ddcf.append(descuento)
            ddcf.append(perpetuidad)
            
    perpetuidad = perpetuidad / sum(ddcf)    
            
    enterprise_value = np.array(ddcf).sum()

    equity = enterprise_value - netdebt

    stock_value = equity / shares
    
    return [stock_value, perpetuidad]

def calcular_waccs(wacc_central, cambio=0.01, n=5):
    wacc_array = wacc_central * (np.linspace(1 - n*cambio, 1 + n*cambio, 2*n + 1))
    return wacc_array

def calcular_waccs2(wacc_central, cambio=0.01, n=2):
    wacc_array = wacc_central + (np.linspace(-n*cambio, n*cambio, 2*n + 1)) 
    return wacc_array

def calcular_gs(g_central, cambio=0.01, n=5):
    g_array = g_central * (np.linspace(1 - n*cambio, 1 + n*cambio, 2*n + 1))
    return g_array

def calcular_gs2(g_central, cambio=0.01, n=2):
    g_array = g_central + (np.linspace(-n*cambio, n*cambio, 2*n + 1))
    return g_array

def dcf_sensitivity_matrix(
    wacc_values, g_values, free_cash_flows, ticker, netdebt, g_to_use
):
    """
    Construye una matriz (DataFrame) con precio/acción para todos los pares (WACC, g).
    Filas = g ; Columnas = WACC
    """
    waccs = np.atleast_1d(np.array(wacc_values, dtype=float))
    gs = np.atleast_1d(np.array(g_values, dtype=float))

    data = np.empty((gs.size, waccs.size), dtype=float)
    for i, gi in enumerate(gs):
        for j, wj in enumerate(waccs):
            data[i, j] = (valuacion_DCF(
                free_cash_flows=free_cash_flows,
                wacc=wj,
                g=gi,
                ticker=ticker, netdebt= netdebt
            )[0]  / valuacion_DCF(free_cash_flows=free_cash_flows, wacc=acciones[ticker].wacc, g=g_to_use, ticker=ticker, netdebt=netdebt) - 1)[0] * 100

    df = pd.DataFrame(
        data,
        index=[f"{gi:.4%}" for gi in gs],
        columns=[f"{wj:.4%}" for wj in waccs],
    )
    df.index.name = "WACC/g"
    df.columns.name = "WACC"
    return df

def dcf_scenarios(
    wacc_values, g_values, free_cash_flows, ticker, netdebt, g_to_use
):
    """
    Construye una matriz (DataFrame) con precio/acción para todos los pares (WACC, g).
    Filas = g ; Columnas = WACC
    """
    waccs = np.atleast_1d(np.array(wacc_values, dtype=float))
    gs = np.atleast_1d(np.array(g_values, dtype=float))

    data = np.empty((gs.size, waccs.size), dtype=float)
    for i, gi in enumerate(gs):
        for j, wj in enumerate(waccs):
            data[i, j] = valuacion_DCF(
                free_cash_flows=free_cash_flows,
                wacc=wj,
                g=gi,
                ticker=ticker, netdebt= netdebt
            )[0]  

    df = pd.DataFrame(
        data,
        index=[f"{gi:.4%}" for gi in gs],
        columns=[f"{wj:.4%}" for wj in waccs],
    )
    df.index.name = "WACC/g"
    df.columns.name = "WACC"
    return df

def crear_df_con_elasticidades(matriz_sensibilidades):
    
    valor_g = float(matriz_sensibilidades.index[math.ceil(len(matriz_sensibilidades.index)/2) - 1].strip('%')) / 100
    valor_wacc = float(matriz_sensibilidades.columns[math.ceil(len(matriz_sensibilidades.columns)/2) - 1].strip('%')) / 100
    
    valores_g = np.array([float(index.strip("%")) for index in matriz_sensibilidades.index]) / 100
    valores_wacc = np.array([float(col.strip("%")) for col in matriz_sensibilidades.columns]) / 100

    variaciones_vi_wacc = np.array(matriz_sensibilidades.iloc[math.ceil(len(matriz_sensibilidades.index)/2)-1, :] / 100)

    variaciones_vi_gs = np.array(matriz_sensibilidades.iloc[: , math.ceil(len(matriz_sensibilidades.index)/2)-1] / 100)

    cambios_pct_wacc = valores_wacc / valor_wacc - 1

    cambios_pct_gs = valores_g / valor_g - 1

    with np.errstate(divide='ignore', invalid='ignore'):
        elasticidades_w = variaciones_vi_wacc / cambios_pct_wacc
        elasticidades_g = variaciones_vi_gs / cambios_pct_gs

    valores_bps = valores_g / 0.0001 - (valor_g / 0.0001)

    valores_bps_wacc = valores_wacc / 0.0001 - (valor_wacc / 0.0001)
    
    valores_bps2 = (valores_g / valor_g - 1) * 100

    valores_bps_wacc2 = (valores_wacc / valor_wacc - 1) * 100

    elasticidades = pd.DataFrame({"Cambio Relativo WACC (%)": valores_bps_wacc,
              'Elasticidades WACC': elasticidades_w,
              "Cambio Relativo g (%)": valores_bps,
              'Elasticidades g': elasticidades_g}).fillna(0)
    
    return elasticidades, valores_bps_wacc, valores_bps, valores_bps2, valores_bps_wacc2

def calcular_desvios(todas_las_matrices):
    stacked = np.stack(todas_las_matrices)
    desvios = stacked.std(axis=0, ddof=1)
    
    return desvios

