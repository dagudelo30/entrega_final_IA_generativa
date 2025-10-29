import os
import uuid
import streamlit as st
import warnings

warnings.filterwarnings("ignore", message=".*HuggingFaceEmbeddings.*")

# ESTO DEBE SER LO PRIMERO
st.set_page_config(page_title="Agente Devoluciones • EcoMarket", page_icon="🛍️", layout="wide")

# =========================
#  Cargar backend una vez
# =========================
@st.cache_resource(show_spinner=False)
def load_bundle(use_sqlite: bool = False):
    import codigo_2 as m
    bundle = m.get_agent_app(use_sqlite=use_sqlite)
    return bundle, m

bundle, mod = load_bundle(use_sqlite=False)
app = bundle["app"]
graph = bundle["graph"]

# =========================
#  Helpers UI
# =========================
def draw_graph(app_):
    """Dibuja el grafo probando ambas rutas (con / sin xray)."""
    try:
        g = app_.get_graph()
        try:
            png = g.draw_mermaid_png()
        except Exception:
            png = app_.get_graph(xray=True).draw_mermaid_png()
        st.image(png, caption="Vista del flujo")
    except Exception as e:
        st.info("No se pudo dibujar el grafo.")
        st.caption(str(e))

def hint_if_order_like(text: str) -> str | None:
    """Si el texto parece un ID (6-12 dígitos) pero no existe en el CSV, devuelve una pista."""
    import re
    m = re.search(r"\b\d{6,12}\b", text or "")
    if not m:
        return None
    candidate = m.group(0)
    try:
        ids = set(str(x) for x in mod.PEDIDOS_DF["order_id"].tolist())
    except Exception:
        ids = set(str(x) for x in mod.PEDIDOS_DF.index.tolist())
    return None if candidate in ids else f"🔎 Nota: detecté `{candidate}`,no está en la base de pedidos. Por favor verifica el número."

def display_state_details(state: dict):
    """Muestra los detalles del estado en formato JSON expandible."""
    if state:
        with st.expander("📊 Estado detallado del agente (JSON)"):
            # Filtrar campos que no queremos mostrar o son demasiado largos
            display_state = {}
            for key, value in state.items():
                if key not in ['messages', 'raw']:  # Excluir campos largos
                    if value is not None:  # Solo mostrar campos con valores
                        display_state[key] = value
            
            st.json(display_state, expanded=False)

# =========================
#  Estado de sesión - SOLO PARA LA UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response_state" not in st.session_state:
    st.session_state.last_response_state = {}

# =========================
#  Sidebar
# =========================
with st.sidebar:
    st.markdown("### Grafo del Agente")
    draw_graph(app)
    
    st.markdown("---")
    st.markdown("### Última Ejecución")
    if st.session_state.last_response_state:
        st.json(st.session_state.last_response_state, expanded=False)
    else:
        st.info("Aún no hay ejecuciones")

# =========================
#  Cabecera + historial
# =========================
st.title("🛍️ Agente de Devoluciones • EcoMarket")
st.caption("💡 **Cada consulta es independiente** - El agente no recuerda conversaciones anteriores")

# Mostrar historial de mensajes (solo para UI, no afecta al agente)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # Mostrar estado detallado solo para respuestas del asistente
        if m["role"] == "assistant" and m.get("state_details"):
            display_state_details(m["state_details"])

# =========================
#  Entrada del usuario
# =========================
user_text = st.chat_input("Escribe tu mensaje (ej. 'Quiero devolver el pedido 2509006')")

if user_text:
    # ========= NUEVO THREAD ID PARA CADA MENSAJE =========
    # Esto fuerza a que cada mensaje sea completamente independiente
    current_thread_id = f"msg-{uuid.uuid4().hex[:8]}"
    
    # Pintar turno usuario
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Pista opcional si parece ID pero no existe
    hint = hint_if_order_like(user_text)
    if hint:
        with st.chat_message("assistant"):
            st.markdown(hint)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": hint,
            "state_details": {"type": "hint", "message": hint}
        })

    # ========= ESTADO COMPLETAMENTE NUEVO PARA CADA MENSAJE =========
    clean_state = {
        "user_message": user_text,
        "conversation_id": current_thread_id,
        # INICIALIZAR TODO EN NONE - ESTADO COMPLETAMENTE FRESCO
        "order_id": None,
        "pedido": None,
        "elegibilidad": None,
        "etiqueta": None,
        "event": None,
        "respuesta": None,
        "messages": []
    }
    
    cfg = {"configurable": {"thread_id": current_thread_id}}

    # ========== EJECUTAR EL AGENTE ==========
    reply = None
    full_state = {}
    
    with st.spinner("🔍 Procesando consulta..."):
        try:
            out_state = app.invoke(clean_state, config=cfg)
            
            # Extraer el estado completo para mostrar
            if hasattr(out_state, 'get') and callable(out_state.get):
                reply = out_state.get("respuesta", "")
                full_state = dict(out_state)  # Convertir a dict completo
            elif hasattr(out_state, "values"):
                state_values = out_state.values
                if hasattr(state_values, 'get') and callable(state_values.get):
                    reply = state_values.get("respuesta", "")
                    full_state = dict(state_values)
                else:
                    reply = str(state_values)
            else:
                reply = str(out_state)
                full_state = {"raw_output": str(out_state)}

            reply = (reply or "").strip() or "Lo siento, no pude generar una respuesta."
            
            # Guardar el estado completo para mostrar en UI
            st.session_state.last_response_state = full_state

        except Exception as e:
            reply = f"⚠️ Error al procesar tu consulta: {str(e)}"
            st.session_state.last_response_state = {"error": str(e)}

    # ========== MOSTRAR RESPUESTA CON DETALLES ==========
    if reply:
        # Guardar mensaje con estado detallado
        st.session_state.messages.append({
            "role": "assistant", 
            "content": reply,
            "state_details": st.session_state.last_response_state
        })
        
        with st.chat_message("assistant"):
            st.markdown(reply)
            # Mostrar detalles expandibles del estado
            display_state_details(st.session_state.last_response_state)

# =========================
#  Controles en footer
# =========================
with st.sidebar:
    st.markdown("---")
    
    # Estadísticas
    st.markdown("### Estadísticas")
    st.metric("Mensajes en sesión", len(st.session_state.messages))
    
    # Botón para limpiar historial (solo UI)
    if st.button("🗑️ Limpiar Historial", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response_state = {}
        st.rerun()
    


# =========================
#  Instrucciones para el usuario
# =========================
with st.expander("💡 Cómo usar el agente"):
    st.markdown("""
    **Ejemplos de consultas:**
    - "Quiero devolver el pedido 2509006"
    - "Consulta para order_id 2509007" 
    - "Estado de devolución para ORD12345"
    - "¿Puedo devolver el producto del pedido 2509008?"
    
    **Características:**
    - Cada consulta es independiente
    - El agente no recuerda conversaciones anteriores
    - Puedes ver el estado interno en JSON
    - El grafo muestra el flujo de decisión
    """)