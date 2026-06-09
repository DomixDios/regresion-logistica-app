import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Regresión Logística", layout="centered")
st.title("Regresión Logística")
st.markdown("Carga un archivo CSV y analiza la relación entre variables.")

# ---------- sesión para guardar el modelo ----------
if "model_fitted" not in st.session_state:
    st.session_state.model_fitted = None
if "result" not in st.session_state:
    st.session_state.result = None
if "X_cols" not in st.session_state:
    st.session_state.X_cols = None
if "y_col" not in st.session_state:
    st.session_state.y_col = None
if "df" not in st.session_state:
    st.session_state.df = None

# ---------- carga CSV ----------
uploaded = st.file_uploader("Sube tu archivo CSV", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state.df = df
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(8), use_container_width=True)
    st.caption(f"Filas: {len(df)} | Columnas: {', '.join(df.columns)}")

    # validar nulos
    if df.isnull().sum().sum() > 0:
        st.warning(
            f"El dataset tiene {df.isnull().sum().sum()} valor(es) nulo(s). "
            "Las filas con nulos serán omitidas."
        )
        df = df.dropna()
        st.session_state.df = df

    cols = list(df.columns)

    st.divider()
    st.subheader("Configuración del modelo")

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Variable dependiente (target 0/1)", cols)
    with col2:
        predictors = st.multiselect(
            "Variables independientes (predictores)",
            [c for c in cols if c != target],
        )

    # ---------- calcular ----------
    if st.button("Calcular Regresión", type="primary"):
        if not predictors:
            st.error("Selecciona al menos una variable independiente.")
            st.stop()

        X = df[predictors].copy()
        y = df[target].copy()

        if not y.dropna().isin([0, 1]).all():
            st.error(
                f"La variable '{target}' debe contener solo valores 0 y 1."
            )
            st.stop()

        # agregar constante
        X = sm.add_constant(X)

        with st.spinner("Calculando regresión logística..."):
            modelo = sm.Logit(y, X)
            try:
                resultado = modelo.fit(disp=False, maxiter=200)
            except Exception as e:
                st.error(f"Error al ajustar el modelo: {e}")
                st.stop()

        st.session_state.model_fitted = modelo
        st.session_state.result = resultado
        st.session_state.X_cols = predictors
        st.session_state.y_col = target

        st.success("Modelo calculado correctamente.")
        st.balloons()

    # ---------- mostrar resultados ----------
    if st.session_state.result is not None:
        resultado = st.session_state.result
        X_cols = st.session_state.X_cols
        target = st.session_state.y_col
        df = st.session_state.df
        X = sm.add_constant(df[X_cols])
        y = df[target]

        pred_prob = resultado.predict(X)
        pred_class = (pred_prob >= 0.5).astype(int)

        st.divider()
        st.subheader("Resultados")

        # ---------- TABLA DE COEFICIENTES ----------
        with st.expander("Coeficientes del modelo", expanded=True):
            coef_df = pd.DataFrame({
                "Variable": resultado.params.index,
                "B (Coeficiente)": resultado.params.values,
                "p-valor": resultado.pvalues.values,
                "Odds Ratio (e^B)": np.exp(resultado.params).values,
            })
            coef_df["Significancia"] = coef_df["p-valor"].apply(
                lambda p: "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            )
            coef_df["B (Coeficiente)"] = coef_df["B (Coeficiente)"].round(4)
            coef_df["p-valor"] = coef_df["p-valor"].round(5)
            coef_df["Odds Ratio (e^B)"] = coef_df["Odds Ratio (e^B)"].round(4)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)
            st.caption("*** p < 0.001  ** p < 0.01  * p < 0.05")

        # ---------- RESUMEN DEL MODELO ----------
        with st.expander("Resumen del modelo", expanded=True):
            col_r1, col_r2, col_r3 = st.columns(3)
            llf = resultado.llf
            llnull = resultado.llnull
            n = resultado.nobs

            # McFadden
            r2_mcf = 1 - llf / llnull
            col_r1.metric("R² McFadden", f"{r2_mcf:.4f}")

            # Cox & Snell
            r2_cs = 1 - np.exp(-2 * (llf - llnull) / n)
            col_r2.metric("R² Cox & Snell", f"{r2_cs:.4f}")

            # Nagelkerke
            r2_nag = r2_cs / (1 - np.exp(2 * llnull / n))
            col_r3.metric("R² Nagelkerke", f"{r2_nag:.4f}")

            # Chi-cuadrado
            chi2 = resultado.llr
            chi2_p = resultado.llr_pvalue
            st.markdown(
                f"**Prueba Chi-cuadrado:** χ² = {chi2:.4f}, "
                f"p = {chi2_p:.6f}  — "
                f"{'Modelo significativo ✅' if chi2_p < 0.05 else 'Modelo no significativo ❌'}"
            )

        # ---------- MATRIZ DE CONFUSIÓN ----------
        with st.expander("Clasificación (umbral 0.5)", expanded=True):
            cm = confusion_matrix(y, pred_class)
            tn, fp, fn, tp = cm.ravel()
            precision = (tp + tn) / cm.sum()

            cm_df = pd.DataFrame(
                cm,
                index=["Real: 0", "Real: 1"],
                columns=["Pred: 0", "Pred: 1"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.metric(
                "Precisión total",
                f"{precision:.2%}",
                help=f"{tp + tn} de {cm.sum()} casos correctos",
            )

        # ---------- CURVA ROC ----------
        with st.expander("Curva ROC", expanded=True):
            fpr, tpr, _ = roc_curve(y, pred_prob)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"AUC = {roc_auc:.4f}",
                    line=dict(color="#4f46e5", width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Aleatorio",
                    line=dict(color="gray", dash="dash"),
                )
            )
            fig.update_layout(
                xaxis_title="1 - Especificidad (Falsos Positivos)",
                yaxis_title="Sensibilidad (Verdaderos Positivos)",
                width=500,
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(x=0.7, y=0.05),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------- CALCULADORA ----------
        st.divider()
        st.subheader("Calculadora de probabilidad")

        if st.session_state.model_fitted is not None:
            X_pred = []
            calc_cols = st.columns(min(3, len(X_cols)))
            for i, col in enumerate(X_cols):
                with calc_cols[i % len(calc_cols)]:
                    c_min = float(df[col].min())
                    c_max = float(df[col].max())
                    val = st.slider(
                        f"{col}",
                        min_value=c_min,
                        max_value=c_max,
                        value=(c_min + c_max) / 2,
                        step=(c_max - c_min) / 100 if c_max != c_min else 1.0,
                    )
                    X_pred.append(val)

            if st.button("Calcular probabilidad"):
                input_df = pd.DataFrame(
                    [[1] + X_pred], columns=["const"] + X_cols
                )
                prob = resultado.predict(input_df)[0]
                st.markdown(
                    f"### Probabilidad: {prob:.2%}"
                )
                st.progress(float(prob))

                st.markdown(
                    f"**Clasificación:** "
                    f"{'Sí (evento ocurre)' if prob >= 0.5 else 'No (evento no ocurre)'}"
                )

else:
    st.info("Sube un archivo CSV para comenzar.")

st.divider()
st.markdown(
    "Hecho con Python · Streamlit · statsmodels · scikit-learn · Plotly",
    help="Proyecto final Machine Learning",
)
