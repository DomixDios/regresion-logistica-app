import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Regresión Logística", layout="centered")
st.title("Regresión Logística")
st.markdown("Carga un archivo CSV y analiza la relación entre variables.")

# ---------- documentación inicial ----------
with st.expander("📖 ¿Cómo funciona esta aplicación?"):
    st.markdown("""
    ### ¿Qué es la regresión logística?

    Es un método estadístico que permite **predecir si algo ocurre o no** (sí/no, 0/1)
    basándose en una o más variables de entrada.

    **Ejemplo con tus datos:** predecir si una persona **compra** (1) o **no compra** (0)
    según su edad, salario, género, estrato e hijos.

    ### ¿Qué necesito?

    - Un archivo **CSV** con una columna que tenga solo **valores 0 y 1** (lo que quieres predecir)
    - Una o más columnas adicionales que servirán como **predictores** (edad, salario, etc.)

    ### ¿Qué voy a obtener?

    | Resultado | ¿Qué me dice? |
    |---|---|
    | **Coeficientes B** | El peso de cada variable en la decisión final |
    | **p-valor** | Si esa variable realmente influye o es solo ruido |
    | **Odds Ratio** | Cuánto cambia la probabilidad al aumentar esa variable |
    | **R²** | Qué tan bien el modelo explica los datos |
    | **Chi-cuadrado** | Si el modelo es mejor que simplemente adivinar |
    | **Matriz de confusión** | Cuántos aciertos y errores tiene el modelo |
    | **Curva ROC / AUC** | Qué tan bien separa el modelo las dos clases |
    | **Calculadora** | Permite probar valores y ver la probabilidad calculada |

    ---
    *Carga tu CSV y configura el modelo en la sección de abajo.*
    """)

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

# ---------- documentación final detallada ----------
st.divider()
with st.expander("📖 Explicación detallada de cada resultado"):
    st.markdown("""
    ---
    ### Coeficientes B

    Cada variable predictora recibe un número llamado **coeficiente B**. Este número indica:

    - **Si B es positivo (+):** al aumentar esa variable, **aumenta** la probabilidad de que ocurra el evento.
    - **Si B es negativo (-):** al aumentar esa variable, **disminuye** la probabilidad de que ocurra el evento.
    - **Si B es cero (0):** esa variable **no tiene efecto**.

    **Fórmula:** Z = B₀ + B₁·X₁ + B₂·X₂ + ... + Bₙ·Xₙ  
    donde B₀ es la constante y B₁, B₂... son los coeficientes de cada predictor.

    ---
    ### p-valor

    El **p-valor** responde a la pregunta: ¿este coeficiente es confiable o podría ser producto del azar?

    | p-valor | Significado |
    |---|---|
    | **p < 0.001** | Altamente significativo (***) |
    | **p < 0.01** | Muy significativo (**) |
    | **p < 0.05** | Significativo (*) — la variable **influye de verdad** |
    | **p ≥ 0.05** | No significativo — la variable **no aporta** al modelo |

    En la tabla de coeficientes, las variables con **p < 0.05** son las que realmente importan.

    ---
    ### Odds Ratio (e^B)

    El **Odds Ratio** se calcula como **e^B** (elevar el número de Euler al coeficiente B).
    Responde: ¿cuántas veces más probable es el evento cuando la variable aumenta una unidad?

    - **OR = 1.5** → por cada unidad que aumenta la variable, la probabilidad aumenta **50%**
    - **OR = 2.0** → por cada unidad, la probabilidad se **duplica**
    - **OR = 0.5** → por cada unidad, la probabilidad se **reduce a la mitad**

    ---
    ### R² (McFadden, Cox & Snell, Nagelkerke)

    Los **R²** miden qué tan bien el modelo explica los datos. Van de **0 a 1**:

    - **0** → el modelo no explica nada (es como adivinar)
    - **1** → el modelo explica todo (perfección, casi nunca pasa)
    - **0.2 – 0.4** → valores típicos para problemas de ciencias sociales
    - **> 0.5** → muy buen modelo

    Se muestran tres versiones porque cada una ajusta el cálculo de forma distinta.
    La más usada en regresión logística es **McFadden**.

    ---
    ### Chi-cuadrado

    La **prueba Chi-cuadrado** compara tu modelo completo contra un modelo que solo tiene la constante
    (es decir, un modelo que siempre predice el valor más frecuente sin usar ninguna variable).

    - **Si p < 0.05** → tu modelo es **significativamente mejor** que adivinar ✅
    - **Si p ≥ 0.05** → tu modelo **no es mejor** que adivinar ❌

    Un Chi-cuadrado significativo es el primer requisito para que el modelo sea útil.

    ---
    ### Matriz de confusión

    Compara lo que el modelo **predijo** contra lo que realmente **ocurrió**,
    usando un **umbral del 50%** (si la probabilidad es ≥ 0.5, se clasifica como "Sí").

    ```
                        Predijo: No    Predijo: Sí
    Real: No    →         TN (Bien)      FP (Falsa alarma)
    Real: Sí    →         FN (Error)     TP (Bien)
    ```

    | Sigla | Significado | ¿Es bueno? |
    |---|---|---|
    | **TN** | True Negative — predijo No y era No | ✅ |
    | **TP** | True Positive — predijo Sí y era Sí | ✅ |
    | **FP** | False Positive — predijo Sí pero era No | ❌ (falsa alarma) |
    | **FN** | False Negative — predijo No pero era Sí | ❌ (se lo perdió) |

    **Precisión total = (TN + TP) / (TN + FP + FN + TP)**

    ---
    ### Curva ROC y AUC

    La **curva ROC** grafica dos métricas para cada posible umbral de clasificación:

    - **Eje Y (Sensibilidad):** qué tan bien detecta los "Sí" verdaderos
    - **Eje X (1 - Especificidad):** cuántos "No" falsamente marca como "Sí"

    La línea gris punteada representa un modelo **aleatorio** (adivinar).

    **AUC (Área Bajo la Curva)** resume la calidad del modelo en un solo número:

    | AUC | Significado |
    |---|---|
    | **0.5** | El modelo adivina (igual que lanzar una moneda) |
    | **0.7 – 0.8** | Aceptable |
    | **0.8 – 0.9** | Bueno |
    | **0.9 – 1.0** | Excelente |
    | **1.0** | Perfecto (casi nunca ocurre) |

    ---
    ### Calculadora de probabilidad

    Después de entrenar el modelo, puedes probar **tus propios valores** moviendo los sliders.

    El cálculo que hace es:

    ```
    1. Z = B₀ + B₁·X₁ + B₂·X₂ + ... + Bₙ·Xₙ
    2. Probabilidad = 1 / (1 + e^(-Z))
    ```

    Donde:
    - **B₀, B₁...** son los coeficientes que el modelo aprendió de tus datos
    - **X₁, X₂...** son los valores que tú eliges en los sliders
    - **e** es el número de Euler (~2.718)

    Si la probabilidad es **≥ 50%** → el modelo predice **"Sí"**  
    Si la probabilidad es **< 50%** → el modelo predice **"No"**

    ---
    ### Fórmula general del modelo

    La regresión logística se define matemáticamente como:

    $$P(Y=1) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\beta_n X_n)}}$$

    Donde **P(Y=1)** es la probabilidad de que el evento ocurra, y los **β** son los coeficientes
    que se estiman mediante el método de **máxima verosimilitud** (el modelo prueba diferentes
    combinaciones de β hasta encontrar las que mejor se ajustan a tus datos).
    """)

st.divider()
st.markdown(
    "Hecho con Python · Streamlit · statsmodels · scikit-learn · Plotly",
    help="Proyecto final Machine Learning",
)
