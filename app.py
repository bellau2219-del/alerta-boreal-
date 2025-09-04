# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="Alerta Boreal", layout="wide")
st.title("🌍 Alerta Boreal — Deshielo del Ártico")

# -----------------------
# 1) Función helper para imágenes locales con fallback
# -----------------------
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
IMG_DIR = BASE_DIR / "images"

def get_image_path(local_name: str, fallback_url: str) -> str:
    if not local_name:
        return fallback_url
    p = IMG_DIR / local_name
    if p.exists():
        return str(p)
    return fallback_url

# -----------------------
# 2) Cargar datos (CSV opcional)
# -----------------------
try:
    datos = pd.read_csv("data/hielo_artico_real.csv")
    if not set(["Año", "Millones_km2"]).issubset(datos.columns):
        raise ValueError("CSV sin columnas 'Año' y 'Millones_km2'")
    datos = datos.sort_values("Año").reset_index(drop=True)
except Exception:
    st.warning("No se encontró data/hielo_artico_real.csv — usando datos de ejemplo.")
    datos = pd.DataFrame({
        "Año": [1980, 1990, 2000, 2010, 2020],
        "Millones_km2": [7.5, 6.8, 5.5, 4.2, 3.5]
    })

# -----------------------
# 3) Modelo de predicción (regresión lineal simple)
# -----------------------
X = datos["Año"].values.reshape(-1, 1)
y = datos["Millones_km2"].values
modelo = LinearRegression()
modelo.fit(X, y)

def hielo_para_anio(anio):
    if anio in datos["Año"].values:
        return float(datos.loc[datos["Año"] == anio, "Millones_km2"].iloc[0])
    else:
        return float(modelo.predict(np.array([[anio]]))[0])

baseline_hielo = float(datos["Millones_km2"].max())

# -----------------------
# 4) Especies (con imágenes locales + fallback)
# -----------------------
especies = {
    "Oso polar": {
        "desc": "Alta dependencia del hielo marino para cazar y reproducirse.",
        "img_local": "oso_polar.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Polar_Bear_-_Alaska_%28cropped%29.jpg",
        "pop_actual": 26000,
        "vulnerabilidad": 0.8
    },
    "Foca anillada": {
        "desc": "Necesita hielo para reproducirse y criar.",
        "img_local": "foca.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/f/f8/Ringed_Seal_in_Alaska_%28cropped%29.jpg",
        "pop_actual": 5000000,
        "vulnerabilidad": 0.6
    },
    "Morsa": {
        "desc": "Usa el hielo como plataforma de descanso; sin él es más vulnerable.",
        "img_local": "morsa.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/3/39/Walrus_-_Kamchatka_-_Russia_%28cropped%29.jpg",
        "pop_actual": 300000,
        "vulnerabilidad": 0.5
    },
    "Narval": {
        "desc": "Cetáceo que depende del hielo marino para alimentarse y migrar.",
        "img_local": "narval.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Narwhal_%28Monodon_monoceros%29.jpg",
        "pop_actual": 80000,
        "vulnerabilidad": 0.4
    }
}

# -----------------------
# 5) Ecosistemas (imágenes locales + fallback)
# -----------------------
ecosistemas = {
    "estable": {
        "img_local": "ecos_estable.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/0/05/Sea_Ice_in_the_Arctic_Ocean.jpg"
    },
    "riesgo": {
        "img_local": "ecos_riesgo.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/6/62/Arctic_sea_ice_melt.jpg"
    },
    "critico": {
        "img_local": "ecos_critico.jpg",
        "img": "https://upload.wikimedia.org/wikipedia/commons/f/f8/Arctic_ocean_melting.jpg"
    }
}

# -----------------------
# 6) Layout con pestañas
# -----------------------
tab_datos, tab_graficas, tab_especies, tab_ecosistema, tab_biodiv = st.tabs([
    "📂 Datos", "📈 Gráficas", "🐾 Especies", "🗺 Ecosistema por año", "🌱 Biodiversidad"
])

# ---------- TAB: Datos ----------
with tab_datos:
    st.subheader("📂 Datos históricos")
    st.dataframe(datos, use_container_width=True)

# ---------- TAB: Gráficas ----------
with tab_graficas:
    st.subheader("📈 Evolución y predicción")
    fig1, ax1 = plt.subplots(figsize=(9,4.5))
    ax1.plot(datos["Año"], datos["Millones_km2"], marker="o", label="Datos reales")
    años_ext = np.arange(int(datos["Año"].min()), 2051)
    ax1.plot(años_ext, [hielo_para_anio(a) for a in años_ext],
             linestyle="--", marker="x", color="red", label="Predicción")
    ax1.set_xlabel("Año"); ax1.set_ylabel("Millones km²")
    ax1.grid(True); ax1.legend()
    st.pyplot(fig1, use_container_width=True)

# ---------- TAB: Especies ----------
with tab_especies:
    st.subheader("🐾 Selecciona una especie")
    especie_sel = st.selectbox("Especie:", list(especies.keys()))
    info = especies[especie_sel]
    img_path = get_image_path(info["img_local"], info["img"])
    st.image(img_path, caption=especie_sel, width=380)
    st.markdown(f"**Descripción:** {info['desc']}")
    st.markdown(f"**Población actual (estimada):** {info['pop_actual']:,}")
    
    año_para_eval = st.slider("Calcula pérdidas para el año:",
                              int(datos["Año"].min()), 2050, int(datos["Año"].max()))
    hielo_año = hielo_para_anio(año_para_eval)
    frac = max(0.0, hielo_año / baseline_hielo)

    # ⚡ Riesgo dinámico
    riesgo_dinamico = info["vulnerabilidad"] * (1 - frac)
    perdidos = round(info["pop_actual"] * riesgo_dinamico)
    restantes = int(info["pop_actual"] - perdidos)

    # Clasificación de riesgo
    if riesgo_dinamico < 0.2:
        riesgo_texto, color = "Bajo 🟢", "green"
    elif riesgo_dinamico < 0.4:
        riesgo_texto, color = "Moderado 🟡", "orange"
    elif riesgo_dinamico < 0.7:
        riesgo_texto, color = "Alto 🟠", "darkorange"
    else:
        riesgo_texto, color = "Crítico 🔴", "red"

    st.markdown("---")
    st.markdown(f"### Resultado estimado para **{año_para_eval}**")
    st.write(f"- Extensión de hielo: **{hielo_año:.2f} M km²**")
    st.write(f"- Individuos perdidos: **{perdidos:,}**")
    st.write(f"- Restantes: **{restantes:,}**")
    st.markdown(f"- Riesgo actual: <span style='color:{color}; font-weight:bold'>{riesgo_texto}</span>", unsafe_allow_html=True)

# ---------- TAB: Ecosistema ----------
with tab_ecosistema:
    st.subheader("🗺 Ecosistema del Ártico según el año")
    año_sel = st.slider("Selecciona un año:", int(datos["Año"].min()), 2050, int(datos["Año"].max()))
    hielo_sel = hielo_para_anio(año_sel)
    frac = max(0.0, hielo_sel / baseline_hielo)

    # Estado del ecosistema
    if hielo_sel > 6:
        estado = "estable"; st.success("Ecosistema estable 🌱")
    elif 4 <= hielo_sel <= 6:
        estado = "riesgo"; st.warning("Ecosistema en riesgo ⚠️")
    else:
        estado = "critico"; st.error("Ecosistema crítico 🚨")

    img_path = get_image_path(ecosistemas[estado]["img_local"], ecosistemas[estado]["img"])
    st.image(img_path, caption=f"Ecosistema {estado}", use_column_width=True)

    # Ubicaciones de especies
    especies_coords = {
        "Oso polar": {"lat": 78, "lon": -20},
        "Foca anillada": {"lat": 75, "lon": 10},
        "Morsa": {"lat": 70, "lon": -160},
        "Narval": {"lat": 76, "lon": -60},
    }

    map_data = []
    for especie, info in especies.items():
        riesgo_dinamico = info["vulnerabilidad"] * (1 - frac)
        perdidos = round(info["pop_actual"] * riesgo_dinamico)
        restantes = max(0, info["pop_actual"] - perdidos)

        # Color por riesgo
        if riesgo_dinamico < 0.2:
            color = [0, 200, 0]
        elif riesgo_dinamico < 0.4:
            color = [255, 255, 0]
        elif riesgo_dinamico < 0.7:
            color = [255, 165, 0]
        else:
            color = [255, 0, 0]

        map_data.append({
            "lat": especies_coords[especie]["lat"],
            "lon": especies_coords[especie]["lon"],
            "especie": especie,
            "restantes": restantes,
            "riesgo": riesgo_dinamico,
            "color": color,
            "size": max(200000, restantes / 50)
        })

    view_state = pdk.ViewState(latitude=75, longitude=0, zoom=1, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=["lon", "lat"],
        get_radius="size",
        get_fill_color="color",
        pickable=True,
    )
    tooltip = {
        "html": "<b>{especie}</b><br/>Restantes: {restantes}<br/>Riesgo: {riesgo:.2f}",
        "style": {"backgroundColor": "white", "color": "black"}
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light", tooltip=tooltip))

# ---------- TAB: Biodiversidad ----------
with tab_biodiv:
    st.subheader("🌱 Biodiversidad actual y futura")

    biodiv_data = {
        "Grupo": ["Mamíferos marinos", "Aves marinas", "Peces", "Invertebrados marinos"],
        "Actual": ["17 especies", "280 especies", "≈ 150 especies", "≈ 200 especies"],
        "2050 (riesgo)": ["12 especies (↓30%)", "210 especies (↓25%)", "120 especies (↓20%)", "170 especies (↓15%)"],
        "2100 (crítico)": ["8 especies (↓50%)", "170 especies (↓40%)", "100 especies (↓35%)", "140 especies (↓30%)"]
    }

    df_biodiv = pd.DataFrame(biodiv_data)
    st.write("Comparación estimada de biodiversidad en distintos escenarios climáticos:")
    st.dataframe(df_biodiv, use_container_width=True)

    st.markdown("---")
    st.info("""
    - **Mamíferos marinos**: osos polares, morsas y narvales dependen fuertemente del hielo y son los más vulnerables.
    - **Aves marinas**: muchas especies migratorias perderán hábitats de anidación.
    - **Peces**: desplazamiento hacia aguas más frías, afectando cadenas alimenticias.
    - **Invertebrados**: sensibles al cambio de temperatura y acidificación del océano.
    """)
