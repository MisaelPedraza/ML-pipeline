import pandas as pd

def rfm(ventas_hist, corte):
    # variables RFM por cliente con historial anterior al corte
    g = ventas_hist.groupby('cliente_id')
    t = pd.DataFrame({
        'recencia_dias': (corte - g['fecha'].max()).dt.days,          # días desde la última compra
        'num_transacciones': g['transaccion'].nunique(),              # compras distintas
        'gasto_total': g['monto'].sum(),                              # gasto acumulado
    })
    return t