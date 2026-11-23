import numpy as np
import pandas as pd

def buscar(df, opciones, obligatoria=True):
    # devuelve la primera columna cuyo nombre contiene alguna de las opciones (sin importar mayúsculas)
    for op in opciones:
        for c in df.columns:
            if op in str(c).lower().strip():
                return c
    if obligatoria:
        raise KeyError(f'No se encontró columna para {opciones}: {list(df.columns)}')
    return None

def a_fecha(serie):
    # convierte a fecha probando formato año-mes-día y día/mes/año; se queda con el que tenga menos vacíos
    f1 = pd.to_datetime(serie, errors='coerce')
    if f1.isna().mean() < 0.01:
        return f1.dt.normalize()
    f2 = pd.to_datetime(serie, errors='coerce', dayfirst=True)
    return (f2 if f2.isna().sum() < f1.isna().sum() else f1).dt.normalize()

def a_numero(serie):
    # convierte a número aunque venga como texto con $ o comas
    if serie.dtype == object:
        serie = serie.astype(str).str.replace(r'[$,\s]', '', regex=True)
    return pd.to_numeric(serie, errors='coerce')

def estandarizar(v, archivo, precio_cat):
    # detecta las columnas de UN archivo y lo lleva al formato común
    c = dict(cliente=buscar(v, ['client_id', 'cliente_id', 'customer', 'cliente', 'client']),
             producto=buscar(v, ['product_id', 'producto_id', 'product', 'producto']),
             fecha=buscar(v, ['date', 'fecha']),
             cantidad=buscar(v, ['units', 'quantity', 'cantidad', 'qty', 'unidades'], obligatoria=False),
             monto=buscar(v, ['sales', 'total', 'amount', 'revenue', 'monto', 'importe'], obligatoria=False),
             transaccion=buscar(v, ['transaction', 'sale_id', 'order', 'ticket', 'transaccion', 'venta_id'], obligatoria=False))
    if c['monto'] in (c['cliente'], c['producto'], c['fecha'], c['transaccion']):
        c['monto'] = None                                                  # evita confundir 'sales_id' con monto
    print(f'{archivo} -> {c}')
    t = pd.DataFrame({'cliente_id': a_numero(v[c['cliente']]),
                      'product_id': a_numero(v[c['producto']]),
                      'fecha': a_fecha(v[c['fecha']])})
    t['cantidad'] = a_numero(v[c['cantidad']]) if c['cantidad'] else 1      # 1 si no hay cantidad
    t['monto'] = a_numero(v[c['monto']]) if c['monto'] else t['cantidad'] * t['product_id'].map(precio_cat)
    t['transaccion'] = v[c['transaccion']].astype(str).values if c['transaccion'] else np.nan
    t['archivo'] = archivo
    return t
 
def limpiar_ventas(ventas_por_archivo, productos):
    precio_cat = productos.set_index("product_id")["static_price"]

    ventas = pd.concat(
        [estandarizar(v, r, precio_cat) for r, v in ventas_por_archivo.items()],
        ignore_index=True,
    )

    n0 = len(ventas)
    ventas = ventas.dropna(subset=['cliente_id', 'product_id', 'fecha'])                 # registros sin llave o fecha
    ventas = ventas[ventas['product_id'].isin(productos['product_id'])]                  # solo productos del catálogo
    ventas = ventas[(ventas['cantidad'] > 0) & (ventas['monto'] > 0)]                    # cantidades y montos válidos
    ventas[['cliente_id', 'product_id']] = ventas[['cliente_id', 'product_id']].astype(int)
    sin_trans = ventas['transaccion'].isna()
    ventas.loc[sin_trans, 'transaccion'] = (ventas.loc[sin_trans, 'cliente_id'].astype(str) + '_' +
                                        ventas.loc[sin_trans, 'fecha'].dt.strftime('%Y%m%d'))   # 1 visita = cliente + día
    ventas = ventas.drop_duplicates(subset=['cliente_id', 'product_id', 'fecha', 'cantidad', 'monto', 'transaccion'])  # duplicados

    return ventas