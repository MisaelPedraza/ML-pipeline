from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

K_SEGMENTOS = 4
COVARIANZA = "spherical"
SEMILLA = 42
VARS_RFM = ["recencia_dias", "num_transacciones", "gasto_total"]

def segmentar(t, k=K_SEGMENTOS, cov=None):
    # Modelo 1: GMM sobre RFM estandarizado + nombre de cada segmento
    cov = cov or COVARIANZA
    scaler = StandardScaler()
    X = scaler.fit_transform(t[VARS_RFM])
    gmm = GaussianMixture(n_components=k, covariance_type=cov, n_init=3, random_state=SEMILLA).fit(X)
    t = t.copy()
    t['cluster'] = gmm.predict(X)
    t['prob_segmento'] = gmm.predict_proba(X).max(axis=1)             # confianza de la asignación
    perfil = t.groupby('cluster')[VARS_RFM].mean()
    inactivo = perfil['recencia_dias'].idxmax()                       # más días sin comprar
    resto = perfil.drop(inactivo)['gasto_total'].sort_values(ascending=False).index
    nombres = {inactivo: 'Inactivos', resto[0]: 'VIP', resto[1]: 'Regulares', resto[2]: 'Ocasionales'}
    t['segmento'] = t['cluster'].map(nombres)
    return t, gmm, scaler, X