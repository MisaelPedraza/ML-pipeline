import numpy as np


def construir_matriz_interacciones(ventas_hist, clientes, productos):
    dias_compra = (
        ventas_hist.groupby(["cliente_id", "product_id"])["fecha"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=clientes, columns=productos, fill_value=0)
    )

    return np.log1p(dias_compra)