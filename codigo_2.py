# --- Standard library
import os
import csv
import uuid
import datetime
import json
import re
import unicodedata
import random
import sqlite3

# --- Typing
from typing import Any, Dict, List, Optional, Iterable, Literal, TypedDict

# --- Third-party
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain / LangGraph
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt.chat_agent_executor import AgentState
import pickle

# === RUTAS PORTABLES Y UTILIDADES DE ARCHIVOS =================================
from pathlib import Path

# Raíz del proyecto (carpeta donde está este archivo)
BASE_DIR: Path = Path(__file__).resolve().parent

# Raíz de datos (por defecto ./data junto a este archivo; configurable por env)
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", BASE_DIR / "data")).resolve()

# Subcarpetas recomendadas
DOCS_DIR: Path         = DATA_DIR / "documents"                 # políticas, csvs, etc.
VSTORE_DIR: Path       = DATA_DIR / "vectorstore" / "faiss_policy"  # índice FAISS persistente
CHECKPOINTS_DIR: Path  = DATA_DIR / "checkpoints"               # checkpoints LangGraph
DATA_LOG_DIR: Path     = DATA_DIR / "data_log"                  # bases SQLite/archivos de log

# Archivos clave
CKPT_DB: Path          = CHECKPOINTS_DIR / "langgraph_ckpt.db"  # checkpoint SQLite para conversaciones
# Si usas un CSV concreto de pedidos, puedes fijarlo por env o dejar un nombre por defecto:
CSV_PEDIDOS_FILE = os.environ.get("CSV_PEDIDOS_FILE", "pedidos_2025-09.csv")
CSV_PEDIDOS_PATH: Path = DOCS_DIR / CSV_PEDIDOS_FILE

# Crear carpetas si no existen
for _p in (DOCS_DIR, VSTORE_DIR, CHECKPOINTS_DIR, DATA_LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

def resolve_path(p: str | Path) -> Path:
    """
    Devuelve una ruta absoluta existente para `p`.
    - Si `p` es absoluta y existe -> se devuelve tal cual.
    - Si es relativa, prueba en este orden: DATA_DIR, BASE_DIR, CWD.
    Lanza FileNotFoundError si no se encuentra.
    """
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return pp
    for candidate in (DATA_DIR / pp, BASE_DIR / pp, Path.cwd() / pp):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontró '{p}'. Probé: "
                            f"{DATA_DIR / pp}, {BASE_DIR / pp}, {Path.cwd() / pp}")

# (Opcional) Política por entorno o "última" por patrón — no cambia tu lógica, solo centraliza acceso
POLICY_FILE   = os.environ.get("POLICY_FILE", None)  # ej: "politicas_devoluciones_es_CO_2025-10.md"
POLICY_PATTERN = os.environ.get("POLICY_PATTERN", "politicas_devoluciones_es_CO_*.md")

def get_policy_paths() -> list[Path]:
    """
    Devuelve una lista de rutas de políticas para tu RAG.
    - Si POLICY_FILE está definido -> esa.
    - Si no, devuelve la *más reciente* que cumpla el patrón en DOCS_DIR.
    No crea índices ni hace chunking; solo resuelve rutas.
    """
    if POLICY_FILE:
        return [resolve_path(DOCS_DIR / POLICY_FILE)]
    candidates = sorted(DOCS_DIR.glob(POLICY_PATTERN))
    if not candidates:
        raise FileNotFoundError(
            f"No encontré archivos que cumplan '{POLICY_PATTERN}' en {DOCS_DIR}"
        )
    return [candidates[-1]]  # la más reciente por ordenación alfabética/fecha en nombre

# (Opcional) Prints de diagnóstico (puedes comentar estas líneas si no las quieres ver)
print(f"📂 BASE_DIR      : {BASE_DIR}")
print(f"📂 DATA_DIR      : {DATA_DIR}")
print(f"📂 DOCS_DIR      : {DOCS_DIR}")
print(f"📂 VSTORE_DIR    : {VSTORE_DIR}")
print(f"🗄️ CKPT_DB       : {CKPT_DB}")
print(f"📄 CSV_PEDIDOS   : {CSV_PEDIDOS_PATH}")

## Funciones para LLM OpenRouter ============================================

def _get_openrouter_api_key() -> str:
    """
    Forma original + fallback a variables de entorno.
    (No cambia la firma ni el comportamiento esperado.)
    """
    try:
        import environ  # opcional, si usas django-environ
        env = environ.Env()
        environ.Env.read_env()
        key = env("OPENROUTER_API_KEY", default=None)
    except Exception:
        key = None
    key = key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY no encontrada. Añádela a tu .env o variables de entorno."
        )
    return key


def get_llm():
    """
    Devuelve un ChatOpenAI apuntando a OpenRouter usando el modelo
    meta-llama/llama-3.3-70b-instruct:free (por defecto).
    Puedes cambiar el modelo/base_url por variables de entorno.
    (Firma y comportamiento intactos.)
    """
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    api_key = _get_openrouter_api_key()

    default_headers = {}
    if os.getenv("OPENROUTER_SITE_URL"):
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
    if os.getenv("OPENROUTER_APP_NAME"):
        default_headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=60,
        max_retries=3,
        default_headers=default_headers or None,
    )
    return llm


# (NUEVA) Versión small para tareas que NO son RAG (redacción, copy, etc.)
def get_llm_small():
    """
    Igual que get_llm(), pero permite un modelo pequeño por env para ahorrar latencia/costo
    en tareas que no requieren tanta capacidad.
    No sustituye a get_llm(): es complementaria y retrocompatible.
    """
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv(
        "OPENROUTER_MODEL_SMALL",
        # elige tu preferido; puedes cambiarlo por env sin tocar código
        "qwen/qwen-2.5-7b-instruct:free"
    )
    api_key = _get_openrouter_api_key()

    default_headers = {}
    if os.getenv("OPENROUTER_SITE_URL"):
        default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL")
    if os.getenv("OPENROUTER_APP_NAME"):
        default_headers["X-Title"] = os.getenv("OPENROUTER_APP_NAME")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=45,          # un poco más agresivo para tareas livianas
        max_retries=2,
        default_headers=default_headers or None,
    )
    return llm

def build_policy_rag(k: int = 4):
    """
    Construye el RAG con tu mismo flujo (chunking semántico) + FAISS persistente.
    Devuelve {"ask": ..., "retriever": ..., "llm": ...} como en tu versión.
    """
    # 1) Documentos
    policy_paths = get_policy_paths()
    docs = [
        Document(
            page_content=Path(p).read_text(encoding="utf-8"),
            metadata={"source": Path(p).name},
        )
        for p in policy_paths
    ]

    # 2) Embeddings BGE-M3
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 3) Chunking semántico
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=65,
        min_chunk_size=100,
    )

    # 4) FAISS persistente
    has_index = any(f.suffix in (".faiss", ".pkl") for f in VSTORE_DIR.glob("*"))
    if has_index:
        vs = FAISS.load_local(str(VSTORE_DIR), embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = chunker.split_documents(docs)
        vs = FAISS.from_documents(chunks, embeddings)
        VSTORE_DIR.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(VSTORE_DIR))

    retriever = vs.as_retriever(search_kwargs={"k": k})

    # 5) LLM vía OpenRouter (¡ESTA LÍNEA FALTABA!)
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or _get_openrouter_api_key()
    os.environ["OPENROUTER_API_KEY"] = openrouter_key
    llm = get_llm()  # ← define el LLM local a la función

    # 6) Prompt base
    system_prompt = (
        "Eres un asistente de políticas de EcoMarket. "
        "Responde con precisión basándote únicamente en el contexto proporcionado."
        "Si la información no está en el contexto, responde que no lo sabes. "
    )
    human_prompt = (
        "Pregunta: {question}\n\n"
        "Contexto recuperado:\n{context}\n\n"
        "Respuesta (con citas):"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])

    # 7) Cadena RAG
    def _format_docs(docs_):
        return "\n\n".join(
            f"{d.page_content}\n[Fuente: {d.metadata.get('source', 'desconocido')}]"
            for d in docs_
        )

    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 8) Helper ask
    def ask(question: str):
        docs_ = retriever.invoke(question)
        answer = rag_chain.invoke(question)
        sources = list({d.metadata.get("source", "desconocido") for d in docs_})
        return {"answer": answer, "sources": sources}

    return {"ask": ask, "retriever": retriever, "llm": llm}




# === CARGA DE BASE DE PEDIDOS (desde DOCS_DIR/documents) ======================
import pandas as pd
from pathlib import Path

# CSV_PEDIDOS_PATH ya quedó definido en el bloque de rutas:
#   CSV_PEDIDOS_FILE = os.environ.get("CSV_PEDIDOS_FILE", "pedidos_2025-09.csv")
#   CSV_PEDIDOS_PATH = DOCS_DIR / CSV_PEDIDOS_FILE

if not CSV_PEDIDOS_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo de pedidos en {CSV_PEDIDOS_PATH}.\n"
        f"Colócalo dentro de la carpeta: {DOCS_DIR}"
    )

# Carga robusta sin dependencias extra (primero UTF-8, fallback Latin-1)
try:
    PEDIDOS_DF = pd.read_csv(CSV_PEDIDOS_PATH, dtype=str, encoding="utf-8").fillna("")
except UnicodeDecodeError:
    PEDIDOS_DF = pd.read_csv(CSV_PEDIDOS_PATH, dtype=str, encoding="latin-1").fillna("")

# Normaliza columnas y asegura índice por order_id
PEDIDOS_DF.columns = [c.lower().strip() for c in PEDIDOS_DF.columns]
if "order_id" in PEDIDOS_DF.columns:
    PEDIDOS_DF.set_index("order_id", inplace=True, drop=False)

# Chequeo ligero (opcional; puedes comentar estas dos líneas)
print("✅ Pedidos cargados:", len(PEDIDOS_DF))
print("🧾 Primeros order_id:", PEDIDOS_DF["order_id"].head(5).tolist())

# === BASE DE DATOS LOCAL (eventos) ============================================
# Usa la carpeta recomendada data/data_log/
DB_PATH: Path = DATA_LOG_DIR / "eventos_devoluciones.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"🗄️ Base de datos local: {DB_PATH}")

# ====================== Utils mejoradas (sin cambiar firmas) ======================
# (Asume que ya importaste: re, json, unicodedata, datetime, uuid, sqlite3, Optional, Dict, Any)
# (Asume que existen: PEDIDOS_DF, rag, llm_basico, get_llm_small() opcional, DB_PATH)

# Regex y cachés precompiladas (rendimiento)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_WS_RE = re.compile(r"\s+")
_RE_ORD_PREFIX = re.compile(r"\bORD[-_ ]?(\d{4,12})\b", re.IGNORECASE)
_RE_NUMERIC    = re.compile(r"\b\d{6,12}\b")

_ORDER_IDS_CACHE = None
def _get_order_ids_cache() -> set[str]:
    """Cachea el conjunto de order_ids para validación O(1)."""
    global _ORDER_IDS_CACHE
    if _ORDER_IDS_CACHE is None:
        if "order_id" in PEDIDOS_DF.columns:
            _ORDER_IDS_CACHE = set(str(x) for x in PEDIDOS_DF["order_id"].tolist())
        else:
            _ORDER_IDS_CACHE = set(str(x) for x in PEDIDOS_DF.index.tolist())
    return _ORDER_IDS_CACHE

def _find_json_blob(s: str) -> str:
    """
    Extrae de forma robusta el PRIMER objeto JSON { ... }.
    Si no encuentra uno, devuelve el string original (como tu versión).
    """
    if not s:
        return s
    s = str(s).strip()
    m = _JSON_OBJ_RE.search(s)
    return m.group(0) if m else s

def _normalize_spaces(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def regex_order_id_guess(text: str) -> Optional[str]:
    """
    Heurística: ORD12345 o un número de 6-12 dígitos (ajusta a tu dataset).
    Valida contra PEDIDOS_DF (cacheado).
    """
    text = _normalize_spaces(text or "")
    ids = _get_order_ids_cache()

    candidates: list[str] = []
    m = _RE_ORD_PREFIX.search(text)
    if m:
        # candidato con prefijo y solo dígitos
        candidates.append(m.group(0).replace(" ", "").replace("_", "").upper())  # p.ej. ORD12345
        candidates.append(m.group(1))  # solo dígitos

    for m2 in _RE_NUMERIC.finditer(text):
        candidates.append(m2.group(0))

    for c in candidates:
        if c in ids:
            return c
    return None

def router_node(state: AgentState) -> AgentState:
    """Valida/extrae order_id. Si falta, pide amablemente con LLM."""
    # 0) si ya hay respuesta pendiente, no hacemos nada (respetar short-circuit)
    if state.get("respuesta"):
        return state

    # 1) si ya viene el order_id, salimos
    if state.get("order_id"):
        return state

    user_text = (state.get("user_message") or "").strip()
    oid = None

    # 2) intento por regex (rápido)
    if user_text:
        oid = regex_order_id_guess(user_text)

    # 3) si falla regex, intentamos con LLM extractor (small)
    if not oid and user_text:
        oid = extract_order_id_llm(user_text)

    if oid:
        state["order_id"] = oid
        return state

    # 4) si no hay order_id, pedimos amablemente
    ask = friendly_ask_for_order_id(user_text)
    state["respuesta"] = ask
    # opcional: podrías dejar una traza para depurar
    # print("[router_node] Falta order_id; pidiendo al usuario.")
    return state


def buscar_pedido_node(state: AgentState) -> AgentState:
    """Busca el pedido por order_id y, si no existe, arma el evento y la respuesta."""
    # 0) si ya hay respuesta pendiente, no hacemos nada
    if state.get("respuesta"):
        return state

    oid = state.get("order_id")
    if not oid:
        # no deberíamos llegar aquí si router_node funcionó; nos cubrimos igual
        state["respuesta"] = "¿Me confirmas tu order_id para poder buscar el pedido? Ejemplo: 2509006"
        return state

    # 1) invocar tool (misma API)
    try:
        pedido = buscar_pedido.invoke({"order_id": oid})
    except Exception as e:
        # mantenemos la semántica: si falla, actúa como “no encontrado”
        # print(f"[buscar_pedido_node] error en buscar_pedido: {e}")
        pedido = {}

    # 2) no encontrado -> mensaje + preparar evento
    if not pedido:
        state["respuesta"] = f"No encontré el pedido `{oid}`. ¿Podrías verificar el ID?"
        state["event"] = {
            "order_id": oid,
            "rma": "",
            "status": "Procesando",  # por defecto si no hay pedido
            "notes": "Pedido no encontrado",
            "reason": "",
            "category": "",
            "customer_name": "",
        }
        return state

    # 3) encontrado -> guardamos y continuamos
    state["pedido"] = pedido
    return state

def extract_order_id_llm(user_text: str) -> Optional[str]:
    """
    Pide al LLM un JSON con {"order_id": "<id o null>"}. No inventar.
    Usa LLM 'small' si está disponible para menor latencia, sin cambiar la firma.
    """
    # Preferir LLM pequeño si existe; si no, usar llm_basico
    llm_for_extraction = globals().get("get_llm_small", None)
    llm_obj = llm_for_extraction() if callable(llm_for_extraction) else globals().get("llm_basico", None)

    prompt = (
        "Eres un asistente de devoluciones. Extrae el order_id del siguiente mensaje si está presente.\n"
        "Formato típico: puede ser 'ORD12345' o solo dígitos como '2509006'.\n"
        "Responde SOLO con un JSON válido exactamente así:\n"
        '{"order_id": "<id>"}\n'
        "Si no puedes extraerlo con alta confianza, responde: {\"order_id\": null}\n\n"
        f"Mensaje del usuario:\n{user_text}\n"
    )
    try:
        raw = llm_obj.invoke(prompt) if llm_obj is not None else ""  # ChatOpenAI retorna texto
        js = _find_json_blob(str(raw))
        data = json.loads(js)
        oid = data.get("order_id")
        if not oid:
            return None
        ids = _get_order_ids_cache()
        return oid if oid in ids else None
    except Exception:
        return None


def friendly_ask_for_order_id(previous_msg: Optional[str] = None) -> str:
    """
    Pide de forma amable el order_id al usuario usando el LLM, breve y claro.
    Usa LLM 'small' si está disponible.
    """
    llm_for_copy = globals().get("get_llm_small", None)
    llm_obj = llm_for_copy() if callable(llm_for_copy) else globals().get("llm_basico", None)

    prompt = (
        "Redacta un breve mensaje amable (máx. 2 líneas) pidiendo el order_id al cliente "
        "para gestionar una devolución. Da un ejemplo del formato (ej. '2509006'). No uses jerga técnica."
    )
    try:
        return _normalize_spaces(str(llm_obj.invoke(prompt))) if llm_obj is not None else \
               "¿Me compartes tu order_id para ayudarte con la devolución? Ejemplo: 2509006"
    except Exception:
        return "¿Me compartes tu order_id para ayudarte con la devolución? Ejemplo: 2509006"


def render_final_reply_with_llm(state: AgentState) -> str:
    """
    Genera respuesta amable y clara según el outcome del flujo.
    Usa LLM 'small' si está disponible (no requiere gran razonamiento).
    """
    llm_for_copy = globals().get("get_llm_small", None)
    llm_obj = llm_for_copy() if callable(llm_for_copy) else globals().get("llm_basico", None)

    pedido = state.get("pedido") or {}
    eleg = state.get("elegibilidad") or {}
    etq = state.get("etiqueta") or {}
    raw_reason = eleg.get("reason", "")
    base = {
        "order_id": state.get("order_id") or "",
        "name": pedido.get("name", ""),
        "status": pedido.get("status", ""),
        "category": pedido.get("category", ""),
        "eligible": bool(eleg.get("eligible", False)),
        "rma": etq.get("rma", ""),
        "label_url": etq.get("label_url", ""),
        "reason": raw_reason,
    }
    prompt = (
        "Eres un asistente de atención al cliente. Redacta una respuesta breve, amable y clara en español.\n"
        "Contexto JSON del caso:\n"
        f"{json.dumps(base, ensure_ascii=False)}\n\n"
        "Reglas:\n"
        "- Si eligible es true: felicita, muestra RMA y la URL de etiqueta en una lista.\n"
        "- Si eligible es false: explica con 1-2 frases la razón, invítalo a revisar la política y ofrece ayuda.\n"
        "- No uses tecnicismos; tono cordial.\n"
        "- No inventes datos que no estén en el JSON."
    )
    try:
        return str(llm_obj.invoke(prompt)).strip() if llm_obj is not None else \
               "⚠️ No se pudo contactar al LLM para redactar la respuesta."
    except Exception:
        # Fallback manual
        if base["eligible"]:
            return (
                "✅ Devolución elegible.\n\n"
                f"- RMA: {base['rma']}\n- Etiqueta: {base['label_url']}\n"
                "¿Necesitas algo más?"
            )
        else:
            reason = base["reason"] or "No cumple con los criterios de la política."
            return f"❌ No es elegible para devolución. Motivo: {reason}"


# Mantén tu decisión: re-usa tu RAG y llm básico
llm_basico = get_llm()
      
llm_small  = get_llm_small() 
rag = build_policy_rag(k=4)   # <- tu función existente (con FAISS persistente)


# ============================= Schemas (sin cambios) ===========================
class BuscarPedidoInput(BaseModel):
    order_id: str = Field(..., description="ID del pedido")

class VerificarElegibilidadInput(BaseModel):
    order_id: str
    status: str
    category: str

class GenerarEtiquetaInput(BaseModel):
    order_id: str
    customer_name: Optional[str] = Field(default="", description="Nombre del cliente")
    category: Optional[str] = Field(default="", description="Categoría del producto")
    status: Optional[str] = Field(default="", description="Estado del pedido/producto")

class RegistrarEventoSQLInput(BaseModel):
    order_id: str
    rma: str
    status: Literal["Procesando","En preparación","En tránsito","Retrasado","Entregado","Intento de entrega fallido","Retenido por aduana","En revisión de pago"]
    notes: Optional[str] = None
    reason: Optional[str] = None
    category: Optional[str] = None
    customer_name: Optional[str] = None


# ============================== Tools (mismas 4) ==============================
@tool("buscar_pedido", args_schema=BuscarPedidoInput)
def buscar_pedido(order_id: str) -> Dict[str, Any]:
    """
    Busca un pedido en el DataFrame PEDIDOS_DF por su order_id.
    Devuelve un dict con información básica del pedido o {} si no existe.
    (Firma y salida idénticas a tu versión.)
    """
    if "order_id" not in PEDIDOS_DF.columns:
        return {}
    # Funciona aunque el índice no esté seteado en order_id
    if order_id in PEDIDOS_DF.index:
        row = PEDIDOS_DF.loc[order_id]
    else:
        m = PEDIDOS_DF[PEDIDOS_DF["order_id"] == order_id]
        if m.empty:
            return {}
        row = m.iloc[0]
    return {
        "order_id": row.get("order_id", ""),
        "name": row.get("customer_name", ""),
        "status": row.get("status", ""),
        "category": row.get("category", ""),
        "raw": row.to_dict(),
    }


@tool("verificar_elegibilidad_producto", args_schema=VerificarElegibilidadInput)
def verificar_elegibilidad_producto(order_id: str, status: str, category: str) -> Dict[str, Any]:
    """
    Usa tu RAG para decidir elegibilidad. Devuelve SIEMPRE {"eligible": bool, "reason": str}.
    Evita recursión; parsea JSON con _find_json_blob y retorna un fallback claro si falla.
    """
    prompt = (
        "Devuelve SOLO un JSON válido con esta forma exacta:\n"
        'Revisa la política de devoluciones y responde:\n'
        'Estado del producto: Solo productos con estado Entregado son elegibles para devolución\n'
        'Revisa las categorias permitidas para devoluciones según la política.\n'
        '{"eligible": true|false, "reason": "<texto detallado>"}\n\n'
        f"INSTRUCCIÓN: Basa tu decisión ÚNICAMENTE en la política de devoluciones,Si falta información en el contexto, asume que no cumple los requisitos:\nORDER_ID: {order_id}\nCATEGORIA: {category}\nESTADO_PRODUCTO: {status}"
    )
    out = rag["ask"](prompt)
    s = _find_json_blob(out.get("answer", "").strip())
    try:
        data = json.loads(s)
        eligible = bool(data.get("eligible", False))
        reason = str(data.get("reason", "")).strip() or out.get("answer", "").strip()
        return {"eligible": eligible, "reason": reason}
    except Exception:
        # Fallback ESTRUCTURADO (evita __wrapped__/recursión)
        return {
            "eligible": False,
            "reason": out.get("answer", "No se pudo interpretar la respuesta del modelo.")
        }


def _rma_id(order_id: str, deterministic: bool = True) -> str:
    """Genera un ID único o determinístico para la devolución (misma firma y salida)."""
    if deterministic:
        base = f"{order_id}"
        rma_core = uuid.uuid5(uuid.NAMESPACE_URL, base).hex[:8].upper()
    else:
        rma_core = uuid.uuid4().hex[:8].upper()
    today = datetime.date.today().strftime("%Y%m%d")
    return f"RMA-{today}-{rma_core}"


@tool("generar_etiqueta_devolucion", args_schema=GenerarEtiquetaInput)
def generar_etiqueta_devolucion(order_id: str,
                                customer_name: str = "",
                                category: str = "",
                                status: str = "") -> Dict[str, Any]:
    """
    Genera un RMA de devolución sin necesidad de SKU.
    Devuelve un texto imprimible y una URL simulada.
    (Misma E/S que tu versión.)
    """
    rma = _rma_id(order_id, deterministic=True)
    label_url = f"https://labels.ecomarket.test/{rma}.pdf"
    label_text = (
        "=== ETIQUETA DE DEVOLUCIÓN ===\n"
        f"RMA: {rma}\n"
        f"Order ID: {order_id}\n"
        f"Cliente: {customer_name or 'N/D'}\n"
        f"Categoría: {category or 'N/D'}\n"
        f"Estado: {status or 'N/D'}\n"
        f"Fecha: {datetime.date.today().isoformat()}\n"
        "===============================\n"
    )
    return {"rma": rma, "label_url": label_url, "label_text": label_text}


def _init_db():
    # Misma tabla; añadimos índices y context manager (no cambia tu API)
    with sqlite3.connect(str(DB_PATH)) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eventos (
                event_id TEXT PRIMARY KEY,
                ts TEXT,
                order_id TEXT,
                status TEXT,
                rma TEXT,
                category TEXT,
                customer_name TEXT,
                reason TEXT,
                notes TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eventos_order ON eventos(order_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_eventos_ts ON eventos(ts)")
        con.commit()

_init_db()

@tool("registrar_evento_sql", args_schema=RegistrarEventoSQLInput)
def registrar_evento_sql(order_id: str,
                         status: str,
                         rma: Optional[str] = None,
                         notes: Optional[str] = None,
                         reason: Optional[str] = None,
                         category: Optional[str] = None,
                         customer_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Registra un evento de devolución en SQLite (persistente y consultable).
    (Mismo contrato de E/S.)
    """
    event = {
        "event_id": str(uuid.uuid4()),
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "order_id": order_id,
        "status": status,
        "rma": rma or "",
        "category": category or "",
        "customer_name": customer_name or "",
        "reason": reason or "",
        "notes": notes or "",
    }
    ok = True
    try:
        with sqlite3.connect(str(DB_PATH)) as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO eventos(event_id, ts, order_id, status, rma, category, customer_name, reason, notes)
                VALUES(:event_id, :ts, :order_id, :status, :rma, :category, :customer_name, :reason, :notes)
            """, event)
            con.commit()
    except Exception as e:
        ok = False
        event["error"] = str(e)

    print("[LOG SQL]", event)
    return {"ok": ok, **event}


# ===================== Grafo del agente: encapsulado ==========================
   
class AgentState(TypedDict):
    order_id: Optional[str]
    pedido: Optional[Dict[str, Any]]
    elegibilidad: Optional[Dict[str, Any]]
    etiqueta: Optional[Dict[str, Any]]
    event: Optional[Dict[str, Any]]
    respuesta: Optional[str]
    messages: List[Dict[str, str]]
    user_message: Optional[str]              # ← texto libre del usuario
    conversation_id: Optional[str]

# ---------- Nodos ----------
def router_node(state: AgentState) -> AgentState:
    """Valida/extrae order_id. Si falta, pide amablemente con LLM."""
    if state.get("order_id"):
        return state

    user_text = (state.get("user_message") or "").strip()

    # 1) intento por regex (rápido)
    oid = regex_order_id_guess(user_text) if user_text else None

    # 2) si falla, intento con LLM extractor
    if not oid and user_text:
        oid = extract_order_id_llm(user_text)

    if oid:
        state["order_id"] = oid
        return state

    # 3) si no hay order_id, pedimos amablemente
    ask = friendly_ask_for_order_id(user_text)
    state["respuesta"] = ask
    # opcional: preparar un evento de "falta dato" (no suelo loguear esto)
    return state


def buscar_pedido_node(state: AgentState) -> AgentState:
    """Busca el pedido por order_id y, si no existe, responde inmediatamente sin registrar evento."""
    if state.get("respuesta"):
        return state

    oid = state.get("order_id")
    if not oid:
        state["respuesta"] = "¿Me confirmas tu order_id para poder buscar el pedido? Ejemplo: 2509006"
        return state

    try:
        pedido = buscar_pedido.invoke({"order_id": oid})
    except Exception as e:
        pedido = {}

    # Pedido no encontrado -> mensaje amigable y TERMINAR FLUJO inmediatamente
    if not pedido:
        state["respuesta"] = f"❌ El número de pedido `{oid}` no se encuentra en nuestra base de datos. Por favor, verifica el número e intenta nuevamente."
        return state  # ← Termina aquí, no continúa con el flujo

    # Pedido encontrado -> continuar con el flujo normal
    state["pedido"] = pedido
    return state

def buscar_pedido_node1(state: AgentState) -> AgentState:
    """Busca el pedido por order_id y, si no existe, arma el evento y la respuesta."""
    if state.get("respuesta"):
        return state

    pedido = buscar_pedido.invoke({"order_id": state["order_id"]})
    if not pedido:
        state["respuesta"] = f"No encontré el pedido `{state['order_id']}`. ¿Podrías verificar el ID?"
        # Prepara el evento para que lo escriba el nodo registrar_evento
        state["event"] = {
            "order_id": state["order_id"],
            "rma": "",
            "status": "Procesando",  # valor por defecto si no hay pedido
            "notes": "Pedido no encontrado",
            "reason": "",
            "category": "",
            "customer_name": "",
        }
        return state

    state["pedido"] = pedido
    return state

def elegibilidad_node(state: AgentState) -> AgentState:
    """Consulta el RAG para decidir elegibilidad; si no elegible, arma evento y respuesta."""
    if state.get("respuesta"):
        return state

    pedido = state.get("pedido", {}) or {}
    status = pedido.get("status", "")
    category = pedido.get("category", "")

    out = verificar_elegibilidad_producto.invoke({
        "order_id": state["order_id"],
        "status": status,
        "category": category,
    })
    state["elegibilidad"] = out

    if not out.get("eligible", False):
        state["respuesta"] = f"❌ No elegible para devolución.\n\nMotivo:\n{out.get('reason','(sin detalle)')}"
        # Prepara evento para el nodo registrar_evento
        state["event"] = {
            "order_id": state["order_id"],
            "rma": "",
            "status": status or "Procesando",
            "notes": "No elegible para devolución",
            "reason": out.get("reason", ""),
            "category": category,
            "customer_name": pedido.get("name", ""),
        }
    return state

def etiqueta_node(state: AgentState) -> AgentState:
    """Genera RMA y etiqueta; arma el evento de éxito."""
    if state.get("respuesta"):
        return state

    pedido = state.get("pedido", {}) or {}
    etq = generar_etiqueta_devolucion.invoke({
        "order_id": state["order_id"],
        "customer_name": pedido.get("name",""),
        "category": pedido.get("category",""),
        "status": pedido.get("status",""),
    })
    state["etiqueta"] = etq

    # Prepara evento de éxito para el nodo registrar_evento
    state["event"] = {
        "order_id": state["order_id"],
        "rma": etq.get("rma",""),
        "status": pedido.get("status","Procesando"),
        "notes": "Etiqueta de devolución generada",
        "reason": (state.get("elegibilidad", {}) or {}).get("reason",""),
        "category": pedido.get("category",""),
        "customer_name": pedido.get("name",""),
    }
    return state

def registrar_evento_node(state: AgentState) -> AgentState:
    """Nodo explícito: escribe en SQLite el evento preparado en state['event']."""
    evt = state.get("event") or {}
    if not evt:
        # Nada que registrar; continúa.
        return state

    try:
        registrar_evento_sql.invoke(evt)
    except Exception as e:
        # No bloquea el flujo por fallo de logging; solo imprime.
        print("[registrar_evento_node] error:", e)
    finally:
        # (Opcional) limpiar el event para no re-loggear en branches siguientes
        # state["event"] = None
        pass
    return state

def responder_node(state: AgentState) -> AgentState:
    """Construye la respuesta final para el usuario con LLM (o fallback)."""
    # Si hay una respuesta previa (p.ej., falta order_id o pedido no encontrado), respóndela
    if state.get("respuesta"):
        return state

    # Genera respuesta amigable basada en el outcome del flujo
    answer = render_final_reply_with_llm(state)
    state["respuesta"] = answer
    return state

# ---------- Condiciones ----------
def falta_order_id(state: AgentState) -> bool:
    return not state.get("order_id")

def pedido_no_encontrado(state: AgentState) -> bool:
    """
    Se usa después de buscar_pedido: 
    - Si ya hay respuesta (significa que no se encontró el pedido)
    - Y el pedido es None/empty
    """
    return (state.get("respuesta") is not None and 
            (state.get("pedido") is None or state.get("pedido") == {}))
    
def pedido_no_encontrado1(state: AgentState) -> bool:
    # Se usa después de buscar_pedido: si ya hay respuesta, fue porque no encontró.
    return state.get("respuesta") is not None and (state.get("pedido") is None)

def no_elegible(state: AgentState) -> bool:
    return bool(state.get("elegibilidad") and not state["elegibilidad"]["eligible"])

def listo_para_log(state: AgentState) -> bool:
    # Si existe un payload de event, hay algo para registrar
    return state.get("event") is not None


from langgraph.graph import StateGraph, END

def build_agent_graph() -> StateGraph:
    """Devuelve el StateGraph con los mismos nodos y transiciones, evitando nombres en conflicto."""
    graph = StateGraph(AgentState)

    # Nodos
    graph.add_node("router", router_node)
    graph.add_node("buscar_pedido", buscar_pedido_node)
    graph.add_node("verificar_elegibilidad", elegibilidad_node)  # antes 'elegibilidad'
    graph.add_node("generar_etiqueta", etiqueta_node)            # antes 'etiqueta'
    graph.add_node("registrar_evento", registrar_evento_node)
    graph.add_node("responder", responder_node)

    # Entry point
    graph.set_entry_point("router")

    # Transiciones
    graph.add_conditional_edges(
        "router",
        lambda s: "falta" if not s.get("order_id") else "ok",
        {"falta": "responder", "ok": "buscar_pedido"},
    )

    #graph.add_conditional_edges(
    #    "buscar_pedido",
    #    lambda s: "log" if pedido_no_encontrado(s) else "ok",
    #    {"log": "registrar_evento", "ok": "verificar_elegibilidad"},
    #)
    
    graph.add_conditional_edges(
    "buscar_pedido",
    lambda s: "not_found" if pedido_no_encontrado(s) else "ok",
    {"not_found": "responder", "ok": "verificar_elegibilidad"},  # Cambiado de "log" a "responder"
    )

    graph.add_conditional_edges(
        "verificar_elegibilidad",
        lambda s: "no" if no_elegible(s) else "si",
        {"no": "registrar_evento", "si": "generar_etiqueta"},
    )

    graph.add_edge("generar_etiqueta", "registrar_evento")
    graph.add_edge("registrar_evento", "responder")
    graph.add_edge("responder", END)

    return graph


def compile_agent_app(graph: StateGraph, use_sqlite: Optional[bool] = None):
    """
    Compila el grafo del agente usando solo MemorySaver (sin SQLite),
    evitando errores de serialización (pickle).
    """
    from langgraph.checkpoint.memory import MemorySaver

    try:
        memory = MemorySaver()
        compiled = graph.compile(checkpointer=memory)
        print("✅ Checkpointer: MemorySaver (sin persistencia en disco)")
        return compiled
    except Exception as e:
        print(f"❌ Error compilando grafo con MemorySaver: {e}")
        compiled = graph.compile()
        print("⚙️ Checkpointer desactivado completamente")
        return compiled

def get_agent_app(use_sqlite: Optional[bool] = None):
    """
    Devuelve un dict con { 'app': compiled, 'graph': graph }.
    Mantiene compatibilidad con tu uso en Streamlit (app.invoke(...)).
    """
    g = build_agent_graph()
    compiled = compile_agent_app(g, use_sqlite=use_sqlite)
    return {"app": compiled, "graph": g}


def run_agent_turn(state: AgentState, thread_id: Optional[str] = None) -> AgentState:
    """
    Helper para ejecutar UN turno del agente desde la UI:
    - Usa `state['conversation_id']` como thread_id si no se pasa explícito.
    - Devuelve el nuevo estado (mismo contrato que tu app actual).
    """
    _thread = thread_id or state.get("conversation_id")
    cfg = {"configurable": {"thread_id": _thread}} if _thread else {}
    return app.invoke(state, config=cfg)  # `app` global, definido abajo


# ------- Instancia global para mantener compatibilidad con tu import en la UI ------
# Ej.: from codigo import app as AGENTE_APP
_graph_bundle = get_agent_app(use_sqlite=None)  # respeta USE_SQLITE_CHECKPOINTER env
app = _graph_bundle["app"]                      # ← CompiledGraph (igual que antes)
graph = _graph_bundle["graph"]                  # ← StateGraph (útil para visualización)

# Ayuda a la UI: si la UI quiere visualizar el grafo vía app.get_graph(xray=True)
# muchos CompiledGraph exponen get_graph; si no, devolvemos el StateGraph.
if not hasattr(app, "get_graph"):
    # Exponer método de conveniencia compatible con la UI
    def _get_graph_fallback(xray: bool = False):
        return graph  # la UI ya sabe dibujarlo con mermaid/graphviz si lo desea
    app.get_graph = _get_graph_fallback  # type: ignore




    
if __name__ == "__main__":
    app = compile_agent_app(build_agent_graph())
    print("✅ Agente compilado y listo para pruebas locales.")
else:
    # Evita que se ejecute al importar (por ejemplo, en test_nodos_basicos.py)
    app = None
    graph = None