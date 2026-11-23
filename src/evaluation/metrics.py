import numpy as np
import pandas as pd


def evaluar_top3(compras_reales, puntajes_por_modelo):
    compras = compras_reales.to_numpy(dtype=bool)
    n_comprados = compras.sum(axis=1)
    evaluables = n_comprados > 0

    if not evaluables.any():
        raise ValueError("No hay clientes con compras en la ventana de evaluación.")

    compras_evaluables = compras[evaluables]
    n_comprados = n_comprados[evaluables]
    resultados = []

    for nombre, puntajes in puntajes_por_modelo.items():
        scores = puntajes.reindex_like(compras_reales).to_numpy()[evaluables]
        top3 = np.argsort(-scores, axis=1, kind="stable")[:, :3]
        aciertos = np.take_along_axis(compras_evaluables, top3, axis=1).sum(axis=1)

        resultados.append({
            "Modelo": nombre,
            "Precisión@3": (aciertos / 3).mean(),
            "Recall@3": (aciertos / n_comprados).mean(),
            "F1@3": (2 * aciertos / (3 + n_comprados)).mean(),
        })

    return pd.DataFrame(resultados).set_index("Modelo"), int(evaluables.sum())