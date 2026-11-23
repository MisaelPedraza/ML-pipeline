import pandas as pd
from sklearn.decomposition import TruncatedSVD


def calcular_puntajes_svd(matriz, n_factores, semilla=42):
    modelo = TruncatedSVD(n_components=n_factores, random_state=semilla)
    factores_cliente = modelo.fit_transform(matriz.to_numpy())
    puntajes = factores_cliente @ modelo.components_

    return pd.DataFrame(puntajes, index=matriz.index, columns=matriz.columns)