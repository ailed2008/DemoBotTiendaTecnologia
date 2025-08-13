import os, re, unicodedata
import streamlit as st
import pandas as pd

# IA generativa (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------- Config UI ----------
st.set_page_config(page_title="Tienda Tecnología", page_icon="📱", layout="wide")
st.title("📱🛒 Chat de Tienda de Tecnología (Demo)")
st.caption("Consulta el inventario de celulares, accesorios y cámaras.")

# ---------- Utils ----------
def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text

@st.cache_data
def load_inventory(path="data/inventario.xlsx"):
    df = pd.read_excel(path)
    for col in ["producto","categoria","descripcion"]:
        df[col] = df[col].astype(str)
    return df

def extract_price_filters(q: str):
    """Detecta rangos como 'menos de 1000', 'hasta 2000', 'entre 5000 y 10000'."""
    qn = normalize(q)
    m_between = re.search(r"entre\s+(\d{2,6})\s+y\s+(\d{2,6})", qn)
    if m_between:
        a, b = sorted([int(m_between.group(1)), int(m_between.group(2))])
        return a, b
    m_max = re.search(r"(menos de|hasta|<)\s*(\d{2,6})", qn)
    if m_max:
        return None, int(m_max.group(2))
    m_min = re.search(r"(mas de|desde|>)\s*(\d{2,6})", qn)
    if m_min:
        return int(m_min.group(2)), None
    return None, None

def search_inventory(df, query, categorias, solo_stock, pmin, pmax):
    df_ = df.copy()
    if categorias:
        df_ = df_[df_["categoria"].str.lower().isin([c.lower() for c in categorias])]
    if solo_stock:
        df_ = df_[df_["stock"] > 0]
    if pmin is not None:
        df_ = df_[df_["precio"] >= pmin]
    if pmax is not None:
        df_ = df_[df_["precio"] <= pmax]

    if query:
        qn = normalize(query)
        mask = (
            df_["producto"].str.lower().str.contains(qn, na=False) |
            df_["categoria"].str.lower().str.contains(qn, na=False) |
            df_["descripcion"].str.lower().str.contains(qn, na=False)
        )
        df_ = df_[mask]

    # orden simple: precio ascendente
    return df_.sort_values(["categoria","precio"], ascending=[True, True])

def products_brief(records, max_items=6):
    items = []
    for r in records[:max_items]:
        items.append({
            "producto": r["producto"],
            "categoria": r["categoria"],
            "precio": int(r["precio"]),
            "stock": int(r["stock"]),
            "descripcion": r["descripcion"]
        })
    return items

# ---------- Datos ----------
df = load_inventory()
min_price, max_price = int(df["precio"].min()), int(df["precio"].max())

# ---------- Sidebar (filtros) ----------
with st.sidebar:
    st.subheader("🔎 Filtros")
    cats = st.multiselect("Categorías", options=sorted(df["categoria"].str.lower().unique()),
                          default=[])
    solo_stock = st.checkbox("Solo disponibles (stock > 0)", value=True)

    pmin_user, pmax_user = st.slider(
        "Rango de precio", min_value=min_price, max_value=max_price,
        value=(min_price, max_price), step=100
    )
    st.divider()
    st.caption("Consejo: también puedes escribir 'menos de 2000' o 'entre 5000 y 10000' en la pregunta.")

# ---------- Input ----------
col1, col2 = st.columns([3,1])
with col1:
    q = st.text_input("Escribe tu pregunta (ej: '¿tienes cargadores USB‑C?', 'celulares menos de 7000?')", key="q")
with col2:
    preguntar = st.button("Consultar", type="primary", use_container_width=True)

# Ajuste por texto (precio detectado en la pregunta)
tmin, tmax = extract_price_filters(q or "")
if tmin is not None: pmin_user = max(pmin_user, tmin)
if tmax is not None: pmax_user = min(pmax_user, tmax)

# ---------- Búsqueda ----------
if preguntar and (q or cats or solo_stock or (pmin_user, pmax_user) != (min_price, max_price)):
    resultados = search_inventory(df, q, cats, solo_stock, pmin_user, pmax_user)

    # Mostrar resultados en tarjetas
    st.subheader("Resultados del inventario")
    if resultados.empty:
        st.warning("No encontré productos con esos criterios. Probá ampliar el rango de precio o quitar filtros.")
    else:
        rows = resultados.to_dict(orient="records")
        for i in range(0, len(rows), 3):
            c1, c2, c3 = st.columns(3)
            for col, row in zip((c1,c2,c3), rows[i:i+3]):
                with col:
                    st.markdown(f"**{row['producto']}**  \n_{row['categoria']}_")
                    st.metric("Precio", f"${int(row['precio']):,}".replace(",", "."), help=row["descripcion"])
                    st.caption(f"Stock: {int(row['stock'])}")

        # -------- IA generativa (opcional si hay API key) --------
        # api_key = st.secrets.get("GOOGLE_API_KEY", "AIzaSyBnFrra0uqqQSdKHskygMN6kGk29QBRSPE")
        api_key = st.secrets["GOOGLE_API_KEY"]
        if genai and api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                brief = products_brief(rows)
                prompt = f"""
Eres un asesor de ventas amable y claro. Responde en español, en tono breve.
Pregunta del cliente: {q}
Inventario relevante (máx 6 items): {brief}
Instrucciones:
- Si hay varias opciones, sugiere 2–3 con pros/contras en viñetas.
- Explica compatibilidad (p.ej. USB‑C, iPhone).
- Si la pregunta menciona precio, respétalo.
- Si no hay coincidencias, sugiere alternativas cercanas.
"""
                resp = model.generate_content(prompt)
                st.divider()
                st.subheader("Respuesta del asesor (IA)")
                st.write(resp.text)
            except Exception as e:
                st.info("No se pudo usar Gemini ahora. Se muestran solo resultados del inventario.")
        else:
            st.info("Agrega tu API Key de Gemini en *Secrets* para respuestas generativas.")
else:
    st.caption("Usá el cuadro de búsqueda y/o filtros de la izquierda, luego presioná **Consultar**.")

# Footer
st.divider()
# st.caption("Demo de portafolio • Datos simulados • Reemplaza el CSV por tu inventario real (sin datos sensibles).")
st.caption("Demo de portafolio")