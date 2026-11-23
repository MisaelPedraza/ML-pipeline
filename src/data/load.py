import pandas as pd

def leer(ruta):
    # lee csv o excel; si el csv no está en UTF-8 lo intenta con latin-1
    if not ruta.lower().endswith('.csv'):
        return pd.read_excel(ruta)
    try:
        return pd.read_csv(ruta, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding='latin-1', low_memory=False)