# ---------------------------
# Imports
# ---------------------------

import pandas as pd       # Manejo de datos tabulares (DataFrames)
import numpy as np        # Cálculo numérico eficiente (vectores/arreglos, álgebra, etc.)
import re                 # Expresiones regulares para búsqueda/normalización de nombres de cuentas
import math               # Funciones matemáticas (p.ej., ceil) usadas en cálculos de elasticidad

# ---------------------------
# Clase de dominio: Accion
# ---------------------------

# Acá se define la clase Accion() para la creación de los objetos que contendrán
# el WACC y la cantidad de acciones de cada compañía. Sirve como contenedor
# simple para lookup por ticker.

class Accion():
    def __init__(self, wacc, shares):
        # wacc: costo promedio ponderado de capital (en términos decimales, p.ej. 0.10 = 10%)
        self.wacc = wacc
        # shares: cantidad de acciones en circulación (en millones)
        self.shares = shares  

# ---------------------------
# Parámetros base por ticker
# ---------------------------

# Diccionario de objetos Accion por ticker.
# Nota: Los WACC aquí definidos se usan como "centro" (caso base) en las matrices
# de sensibilidad; las acciones se usan para pasar de equity value a precio por acción.        
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

# ---------------------------
# Limpieza y extracción (4 funciones)
# ---------------------------

# Las siguientes 4 funciones se encargan de la limpieza y extracción de datos
# de los Excels de Bloomberg con las proyecciones cargados en la aplicación.
# El resultado final es un DataFrame limpio con las cuentas relevantes para realizar
# la valuación por DCF.

def limpiar_excel(excel):
    """
    Toma el Excel crudo exportado desde Bloomberg y devuelve:
      - excel_limpio: DataFrame con años como columnas y cuentas como índice.
      - ticker: str detectado de la primera celda (asumido formato estándar).
    Pasos:
      1) Mueve la primera columna a índice (allí viene un header con ticker/etiquetas).
      2) Quita filas/columnas de encabezado redundante.
      3) Mantiene sólo columnas de años y estandariza nombres a 'YYYY'.
    """
    # La primera columna suele contener etiquetas; la mandamos a índice
    excel.index = excel.iloc[:,0]
    # Del mismo encabezado inferimos el ticker (primera "palabra" de la primera fila)
    ticker = excel.index[0].split()[0]
    # Eliminamos la primera columna (ya la usamos como índice)
    excel_limpio = excel.drop(excel.columns[0], axis=1)
    # Removemos fila de título adicional
    excel_limpio = excel_limpio.drop(excel_limpio.index[0])
    # La primera fila remanente contiene headers de columna → pasamos a headers reales
    excel_limpio.columns = excel_limpio.iloc[0]
    # Quitamos esa fila de headers que ya reasignamos
    excel_limpio = excel_limpio.drop(excel_limpio.index[0])
    # Eliminamos columnas no numéricas iniciales (metadatos típicos en exportes)
    excel_limpio = excel_limpio.drop(excel_limpio.iloc[:,0:3], axis=1)
    
    # Nos quedamos con los 4 primeros caracteres de cada header como año 'YYYY'
    anios = [anio[:4] for anio in list(excel_limpio.columns)]
    excel_limpio.columns = anios
    
    # Devolvemos DataFrame con cuentas en índice y años en columnas, y el ticker inferido
    return pd.DataFrame(excel_limpio), ticker

def extraer_partidas(excel_limpio):
    """
    Filtra del DataFrame limpio las filas (cuentas) relevantes para FCFF:
    - Deuda (Total/Net/ST/LT), Cash/Equivalents, Tax Rate, Revenue, Operating Margin,
      Depreciation & Amortization, CAPEX, Current Assets/Liabilities.
    Usa un patrón regex robusto para capturar variantes de nomenclatura.
    Retorna 'partidas' como DataFrame float para facilitar cálculos posteriores.
    """
    # Lista de patrones (regex) que cubren nomenclaturas alternativas comunes en Bloomberg
    patterns = [
        r'^(?!.*\b(?:Re)?Payments\b).*(?:Total Debt$|Debt$|Net(?=.*\bdebt\b))', # Deuda Total / Neta
        r'Equivalents\s\w*\sShort|\bEquivalents\b.*\bMarketable\b|\bCash\b.*Marketable|\bCash\b.*Short-term|\bCash\b.*Equivalents$', # Caja y Equivalentes
        r'^\s*?Tax Rate\s\(\%\)', # Tasa Impositiva %
        r'^\s*?Revenue$|^\s*?Net Revenue', # Ventas
        r'^\s*Operating Margin\s\(\%\)', # Margen Operativo %
        r'(?:Depreciation\s*(?:&|and)\s*Amortization|D\s*&\s*A)', # D&A
        r'(?:Capital\s*Expenditures?$|CAPEX|Capex)', # CAPEX 
        r'^\s*Current Assets$', # Activo Corriente
        r'^\s*Current Liabilities$' # Pasico Corriente
    ]

    # Compilamos un patrón OR con flags case-insensitive
    pat = re.compile('|'.join(patterns), flags=re.IGNORECASE)

    # Creamos máscara booleana sobre el índice (nombres de cuentas)
    mask = excel_limpio.index.astype(str).str.contains(pat, regex=True, na=False)
    partidas = excel_limpio.loc[mask]

    # Normalizamos espacios
    partidas.index = partidas.index.str.strip()
    # Pasamos a float (las columnas son años); filas son cuentas
    partidas = partidas.astype(float)
    
    return partidas

def normalizar_indice(nombre):
    """
    Recibe un nombre de cuenta (string) y devuelve un alias estándar
    usado internamente en el pipeline (e.g., 'Revenue' → 'Ventas').
    Permite consolidar variantes de nombres provenientes del Excel.
    """
    # Normalizamos espacios y casing
    nombre = nombre.strip().lower()

    # Mapeo de patrones → etiqueta estándar que usa el pipeline DCF
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

    # Buscamos el primer patrón que matchee y devolvemos la etiqueta definida
    for patron, nuevo in patrones.items():
        if re.search(patron, nombre, re.IGNORECASE):
            return nuevo
    
    # Si no encontramos patrón, devolvemos el nombre original (ya normalizado en casing)    
    return nombre  

def limpiar_partidas(partidas):
    """
    Post-procesa 'partidas' para:
      - Eliminar filas completamente nulas.
      - Rellenar NaN con 0 (evita interrupciones posteriores).
      - Quitar duplicadas y resolver colisiones (si hay multi-entradas por misma cuenta,
        conservar la fila con mayor suma como 'representativa').
    """
    # Quitamos filas totalmente vacías (todas las columnas NaN)
    partidas = partidas[~partidas.isnull().all(axis=1)] 
    # Relleno defensivo: NaN → 0. Evita errores en aritmética posterior.
    partidas = partidas.fillna(0)
    # Eliminamos duplicados exactos
    partidas = partidas.drop_duplicates()

    # En ocasiones puede quedar un índice duplicado con frames anidados
    cuentas = partidas.index

    # Resolución de conflictos: si por alguna razón loc[cuenta] devuelve DataFrame,
    # tomamos la fila con mayor suma (proxy simple de "entrada principal").
    for cuenta in cuentas:
        if type(partidas.loc[cuenta]) == pd.core.frame.DataFrame:
            operando = partidas.loc[cuenta]
            fila_max = operando[operando.sum(axis=1) == operando.sum(axis=1).max()]
            # Eliminamos las apariciones anteriores…
            partidas = partidas.drop(cuenta)
            # …y dejamos una sola fila con el nombre de la cuenta   
            partidas.loc[cuenta] = fila_max.loc[cuenta]
            
    return partidas

# ---------------------------
# Cálculo de Net Debt (1 función)
# ---------------------------

# La siguiente función se encarga de utilizar el dataframe anterior ya sea para extraer 
# la deuda neta directamente si la partida existe en los datos y, en caso contrario, calcularla
def get_netdebt(partidas):
    """
    Devuelve la Deuda Neta (Net Debt) usando prioridad:
      1) Si existe 'Net Debt', usarla.
      2) Si existe 'Total Debt' y 'Cash', Net = Total - Cash.
      3) Si existen 'ST Debt' + 'LT Debt' y 'Cash', Net = ST + LT - Cash.
      4) Caso mínimo: 'LT Debt' - 'Cash'.
    Se toma el primer año (columna 0) como referencia de deuda neta para el cálculo de equity.
    """
    # Caso ideal: viene reportada la Net Debt
    if 'Net Debt' in partidas.index:
        netdebt = float(partidas.loc['Net Debt', partidas.columns[0]])
        return netdebt
    # Siguiente mejor: Total Debt y Cash
    elif 'Total Debt' in partidas.index:
        netdebt = float(partidas.loc['Total Debt', partidas.columns[0]] - partidas.loc['Cash', partidas.columns[0]])
        return netdebt 
    # Alternativa: ST + LT - Cash
    elif 'ST Debt' in partidas.index and 'LT Debt' in partidas.index:
        netdebt = float(partidas.loc['ST Debt', partidas.columns[0]] + partidas.loc['LT Debt', partidas.columns[0]] - partidas.loc['Cash', partidas.columns[0]]) 
        return netdebt
    # Fallback mínimo
    else:
        netdebt = float(partidas.loc['LT Debt', partidas.columns[0]] - partidas.loc['Cash', partidas.columns[0]]) 
        return netdebt

# ---------------------------
# Validación/completado y derivación de partidas operativas (2 funciones)
# ---------------------------    

# Las siguientes 2 funciones se encargan de:
#  Función: completar_partidas()
#  - Realizar una validación de los datos númericos que se encuentran el dataframe con las proyecciones.
#    En caso de que hubiese celdas en el excel con valores inválidos que pudieran interrumpir el código los reemplaza para evitar errores.

def completar_partidas(partidas):
    for cuenta in partidas.index:
        # Si hay algún 0 en la cuenta, estimamos por proporción a Ventas
        if partidas.loc[cuenta].eq(0).any():
            # Proporción cuenta/Ventas por columna (año)
            porcentaje = partidas.loc[cuenta] / partidas.loc['Ventas']
            # Promedio de proporciones excluyendo ceros (evita sesgo a la baja)
            porcentaje_medio = porcentaje[porcentaje != 0].mean()
            # Ubicamos en qué años la cuenta vale 0
            resultados = partidas.loc[cuenta][partidas.loc[cuenta] == 0]
            # Traemos las ventas de esos mismos años
            ventas = partidas.loc['Ventas', resultados.index]
            # Reemplazamos 0 por Ventas * proporción promedio
            partidas.loc[cuenta, partidas.loc[cuenta] == 0] = ventas * porcentaje_medio
            
    return partidas

#  Función: calcular_nuevas_partidas()
#  - La función completa el dataset financiero generando las partidas operativas necesarias para calcular el FCFF. En particular, 
#   estima el Capital de Trabajo Neto (WK), su variación interanual (Change WK), el beneficio operativo neto después de impuestos (NOPAT) 
#   y finalmente el Free Cash Flow to the Firm (FCFF). A partir de estas nuevas variables, la función devuelve un DataFrame listo 
#   para ser utilizado en las etapas posteriores del modelo DCF.

def calcular_nuevas_partidas(partidas):
    """
    Completa el dataset financiero con partidas clave para FCFF:
      - WK (Working Capital operativo) = (Current Assets - Cash) - Current Liabilities
      - Change WK = Δ interanual del WK (uso .diff() por columnas)
      - NOPAT = Ventas * (Opmg %) * (1 - Tax %)
      - FCFF  = NOPAT + D&A + CAPEX - Change WK
    Nota: Por convención de este pipeline, CAPEX entra con su signo contable (suele ser negativo).
    """
    # Capital de trabajo operativo (excluye efectivo)
    partidas.loc['WK'] = (partidas.loc['Current Assets'] - partidas.loc['Cash']) - partidas.loc['Current Liabilities']
    # Variación interanual del WK (por columnas: años)
    partidas.loc['Change WK'] = partidas.loc['WK'].diff()
    # Beneficio operativo después de impuestos (Opmg y Tax vienen en %)
    partidas.loc['NOPAT'] = partidas.loc['Ventas'] * (partidas.loc['Opmg'] / 100) * (1-partidas.loc['Tax']/100)
    # FCFF no apalancado: suma D&A, agrega CAPEX (negativo si es inversión) y descuenta ΔWK
    partidas.loc['FCFF'] = partidas.loc['NOPAT'] + partidas.loc['D&A'] + partidas.loc['CAPEX'] - partidas.loc['Change WK']

    return partidas

# ---------------------------
# Selección del año de corte para estabilidad de FCFF (1 función)
# ---------------------------

# La función de abajo se encarga de elegir el año de corte en el cual se estabiliza el FCFF, 
# usa como parametro que los flujos de los últimos 4 años no sean negativos
def elegir_ultimo_fcff_estable(df, fila_fcff="FCFF", k_pos=4):
    """
    Devuelve:
      - anio_corte: año (columna) hasta el cual hay k_pos FCFF positivos consecutivos.
      - serie_fcff_cortada: serie FCFF hasta anio_corte inclusive.

    Lógica:
      - Revisa la fila 'FCFF' (como float).
      - Construye una racha de positivos (rolling) de largo k_pos.
      - Si existe alguna ventana completa de positivos, toma el último año de esa ventana como corte.
      - Si no hay 4 seguidos, usa el último FCFF positivo (si existe); si no hay ninguno, levanta error.
    """
    # Obtenemos la serie de FCFF (aseguramos dtype float)
    fcff = df.loc[fila_fcff].astype(float)  
    # Vector 1/0 de positivos
    positivos = (fcff > 0).astype(int)
    # Racha de positivos de largo k_pos    
    racha_pos = positivos.rolling(k_pos, min_periods=k_pos).sum() == k_pos
    # Máscara: FCFF>0 y racha completa
    mascara = (fcff > 0) & racha_pos 

    # Índices (años) candidatos donde la condición se cumple
    candidatos = mascara[mascara].index
    if len(candidatos) > 0:
        anio_corte = candidatos[-1]
    else:
        # Fallback: último año con FCFF>0
        pos = (fcff > 0)
        if pos.any():
            anio_corte = pos[pos].index[-1]
        else:
            # Sin FCFF positivos: no hay base para perpetuidad → revisar supuestos
            raise ValueError("No hay ningún FCFF positivo; revisá supuestos.")
        
    # Cortamos la serie hasta el año de corte
    serie_fcff_cortada = fcff.loc[:anio_corte]
    return anio_corte, serie_fcff_cortada

# ---------------------------
# Valuación DCF por acción (1 función)
# ---------------------------

# Esta función se encarga de realizar la valuación por flujos de caja descontados y devolver el valor resultante.
def valuacion_DCF(wacc, g, free_cash_flows, ticker, netdebt):
    """
    Calcula el precio por acción vía FCFF:
      1) Descuenta cada FCFF al WACC.
      2) En el último año explícito agrega valor terminal (perpetuidad creciente con tasa g).
      3) Suma → Enterprise Value (EV); resta Net Debt → Equity Value.
      4) Divide por 'shares' informadas para el ticker → Precio por acción.
    Devuelve [stock_value, perpetuidad] donde el segundo
    elemento es la fracción de la perpetuidad sobre el total descontado (ratio, no %).

    Notas:
    - Se asume g < wacc para asegurar convergencia de la perpetuidad (criterio estándar DCF).
    - 'free_cash_flows' debe estar en el orden temporal correcto (t=1...n).
    - 'acciones[ticker].shares' debe ser consistente en unidades con el equity.
    """    
    # Acciones en circulación del ticker
    shares = acciones[ticker].shares

    # Acumulador de descuentos descontados (incluye flujos explícitos y luego la perpetuidad)
    ddcf = []

    # Recorremos flujos; el último índice recibe además la perpetuidad
    for año, cashflow in enumerate(free_cash_flows):
        if año != len(free_cash_flows)-1:
            # Descuento del flujo explícito (año+1 porque enumerate arranca en 0)
            descuento = cashflow / (1+wacc)**(año+1)
            ddcf.append(descuento)
        else:
            # Último flujo explícito
            descuento = cashflow / (1+wacc)**(año+1)
            # Valor terminal (perpetuidad creciente): FCFF_(n) * (1+g) / (WACC - g), traído a t=0    
            perpetuidad = (cashflow * (1+g)) / (((1+wacc)**(año+1)) * (wacc - g))
            # Agregamos ambos al vector de descuentos
            ddcf.append(descuento)
            ddcf.append(perpetuidad)

    # Relación (proporción) de la perpetuidad respecto del total descontado (insight de sensibilidad)
    perpetuidad = perpetuidad / sum(ddcf)    

    # Enterprise value: suma de flujos descontados
    enterprise_value = np.array(ddcf).sum()
    # Equity value: EV - Deuda Neta
    equity = enterprise_value - netdebt
    # Precio por acción
    stock_value = equity / shares
    
    return [stock_value, perpetuidad]

# ---------------------------
# Generadores de rangos de valores de WACC y g (4 funciones)
# ---------------------------

# Las próximas 4 funciones se encargan de calcular el rango de valores a utilizar en las
# valuaciones, tanto los valores para el WACC como para la g.
# Las funciones con sufijo "2" arman rangos con cambios absolutos (bps); el resto, cambios relativos (%).
def calcular_waccs(wacc_central, cambio=0.01, n=5):
    """
    Array con cambios relativos en torno al WACC central: desde (1-n*cambio) hasta (1+n*cambio),
    equiespaciada (2n+1 puntos). Ej.: wacc=10%, cambio=1%, n=5 → 10%±5%.
    """
    wacc_array = wacc_central * (np.linspace(1 - n*cambio, 1 + n*cambio, 2*n + 1))
    return wacc_array

def calcular_waccs2(wacc_central, cambio=0.01, n=2):
    """
    Array con cambios absolutos (±bps) en torno al WACC central: desde -n*cambio hasta +n*cambio,
    equiespaciada y sumada al central. Ej.: wacc=10%, cambio=0.01, n=2 → 8%,9%,10%,11%,12%.
    """
    wacc_array = wacc_central + (np.linspace(-n*cambio, n*cambio, 2*n + 1)) 
    return wacc_array

def calcular_gs(g_central, cambio=0.01, n=5):
    """
    Array con cambios relativos para la tasa g: igual idea que calcular_waccs pero aplicada a g.
    Asegurar luego que g < WACC al usarla en perpetuidad.
    """
    g_array = g_central * (np.linspace(1 - n*cambio, 1 + n*cambio, 2*n + 1))
    return g_array

def calcular_gs2(g_central, cambio=0.01, n=2):
    """
    Array con cambios absolutos para g (±bps) alrededor del valor central.
    """
    g_array = g_central + (np.linspace(-n*cambio, n*cambio, 2*n + 1))
    return g_array

# ---------------------------
# Matrices de sensibilidad (2 funciones)
# ---------------------------

# El código de a continuación se encarga de la construcción de la matriz de sensibilidades,
# dicha matriz muestra las variaciones porcentuales en torno al precio calculado en el escenario base.
def dcf_sensitivity_matrix(wacc_values, g_values, free_cash_flows, ticker, netdebt, g_to_use):
    """
    Construye una matriz (DataFrame) con variación % del precio/acción respecto del caso base
    para todos los pares (WACC, g). Convención:
      - Filas = g
      - Columnas = WACC

    Detalles:
      - El caso base usa wacc=acciones[ticker].wacc y g=g_to_use.
      - Cada celda calcula: (Precio_escenario / Precio_base - 1) * 100
    """
    # Normalizamos a arrays 1D
    waccs = np.atleast_1d(np.array(wacc_values, dtype=float))
    gs = np.atleast_1d(np.array(g_values, dtype=float))

    # Pre-alocamos matriz numérica
    data = np.empty((gs.size, waccs.size), dtype=float)
    
    # Para cada combinación (g_i, w_j) valuamos el precio y lo comparamos con el base
    for i, gi in enumerate(gs):
        for j, wj in enumerate(waccs):
            data[i, j] = (valuacion_DCF(
                free_cash_flows=free_cash_flows,
                wacc=wj,
                g=gi,
                ticker=ticker, netdebt= netdebt
            )[0]  / valuacion_DCF(free_cash_flows=free_cash_flows, wacc=acciones[ticker].wacc, g=g_to_use, ticker=ticker, netdebt=netdebt) - 1)[0] * 100

    # Index/columns como % legibles con 4 decimales
    df = pd.DataFrame(
        data,
        index=[f"{gi:.4%}" for gi in gs],
        columns=[f"{wj:.4%}" for wj in waccs],
    )
    df.index.name = "g/WACC"
    df.columns.name = "WACC"
    return df

# La siguiente función construye la matriz de sensibilidad pero mostrando
# rangos de precios (valor por acción) en lugar de variaciones porcentuales.
def dcf_scenarios(wacc_values, g_values, free_cash_flows, ticker, netdebt, g_to_use):
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

# ---------------------------
# Elasticidades locales (1 función)
# ---------------------------

# La función crear_df_con_elasticidades() se encarga de crear la tabla con las
# elasticidades hacia el WACC/g a partir de una matriz de sensibilidades (variaciones %).
def crear_df_con_elasticidades(matriz_sensibilidades):
    """
    A partir de la matriz de sensibilidades (% variación vs caso base), calcula elasticidades
    locales en los ejes WACC y g. Procedimiento:
      - Identifica el centro de la rejilla (fila y columna medio).
      - Extrae la fila "central" (variaciones al mover WACC, con g fijo) y la columna "central"
        (variaciones al mover g, con WACC fijo).
      - Construye cambios relativos en WACC y g respecto al centro.
      - Elasticidad = (ΔValor/Valor) / (ΔVariable/Variable) ≈ (variación %) / (cambio relativo).
    Devuelve:
      - DataFrame con columnas: Cambio Relativo WACC (%), Elasticidades WACC,
                                Cambio Relativo g (%),   Elasticidades g
      - y cuatro arrays auxiliares con "pasos" relativos (promedios/bps) usados.
    """
    # Centro de la rejilla (indices son strings "%"): convertimos a float decimal
    valor_g = float(matriz_sensibilidades.index[math.ceil(len(matriz_sensibilidades.index)/2) - 1].strip('%')) / 100
    valor_wacc = float(matriz_sensibilidades.columns[math.ceil(len(matriz_sensibilidades.columns)/2) - 1].strip('%')) / 100

    # Vectores numéricos de los ejes (en decimales)
    valores_g = np.array([float(index.strip("%")) for index in matriz_sensibilidades.index]) / 100
    valores_wacc = np.array([float(col.strip("%")) for col in matriz_sensibilidades.columns]) / 100

    # Variaciones del valor (como fracción, no %) a lo largo de fila/columna centrales
    variaciones_vi_wacc = np.array(matriz_sensibilidades.iloc[math.ceil(len(matriz_sensibilidades.index)/2)-1, :] / 100)
    variaciones_vi_gs = np.array(matriz_sensibilidades.iloc[: , math.ceil(len(matriz_sensibilidades.index)/2)-1] / 100)

    # Cambios relativos de las variables (ΔX/X) respecto del centro
    cambios_pct_wacc = valores_wacc / valor_wacc - 1
    cambios_pct_gs = valores_g / valor_g - 1

    # Elasticidades: (ΔV/V) / (ΔX/X). Manejo numérico seguro para divisiones por 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        elasticidades_w = variaciones_vi_wacc / cambios_pct_wacc
        elasticidades_g = variaciones_vi_gs / cambios_pct_gs

    # Escalas auxiliares en "pasos" relativos (bps y % relativos alrededor del centro)
    valores_bps_avg = valores_g / 0.0001 - (valor_g / 0.0001)
    valores_bps = (valores_g / valor_g - 1) * 100

    valores_bps_wacc_avg = valores_wacc / 0.0001 - (valor_wacc / 0.0001)
    valores_bps_wacc = (valores_wacc / valor_wacc - 1) * 100

    # Tabla consolidada de elasticidades y cambios relativos
    elasticidades = pd.DataFrame({"Cambio Relativo WACC (%)": valores_bps_wacc,
              'Elasticidades WACC': elasticidades_w,
              "Cambio Relativo g (%)": valores_bps,
              'Elasticidades g': elasticidades_g}).fillna(0)
    
    return elasticidades, valores_bps_wacc_avg, valores_bps_avg, valores_bps, valores_bps_wacc

# ---------------------------
# Desvíos entre matrices (1 función)
# ---------------------------

# La siguiente función se encarga de calcular los desvíos, puede ser aplicada a cualquier conjunto
# de matrices con resultados. En este caso puntual se uso para calcular el desvío de las variaciones
# de precio entre compañías (p.ej., std cell-by-cell sobre una pila de matrices).

def calcular_desvios(todas_las_matrices):
    """
    Recibe una lista/iterable de matrices (2D) numéricas de igual forma (shape).
    Apila en un arreglo 3D y calcula el desvío estándar muestral (ddof=1) en el eje de "empresas".
    Devuelve una matriz 2D de desvíos (misma shape que las matrices de entrada).
    """    
    # Apilamos en un tensor (n_empresas, n_filas, n_cols)
    stacked = np.stack(todas_las_matrices)
    # Desvío muestral over empresas (ddof=1)
    desvios = stacked.std(axis=0, ddof=1)
    
    return desvios