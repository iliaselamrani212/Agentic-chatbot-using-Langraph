"""Front Streamlit : streaming + conversations multiples (threads) + historique SQLite
+ affichage UNIQUEMENT des réponses de l'IA + outils utilisés affichés en temps réel
+ upload de documents PDF pour le RAG + validation humaine (HITL)."""
import os
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from backend import chat, retrieve_all_threads, ingest_rag_document


# =========================== Config ===========================
DEBUG = True   # affiche dans le terminal ce que renvoie le stream (mettre False ensuite)

TOOL_LABELS = {
    "tavily_search": "🔎 Recherche web (Tavily)",
    "get_stock_price": "📈 Prix d'une action (Alpha Vantage)",
    "rag_tool": "📄 Recherche dans les documents (RAG)",
    "purchase": "🛒 Achat d'actions (validation humaine)",
}

UPLOAD_DIR = "uploaded_docs"            # dossier où les PDF uploadés sont enregistrés
DOC_MARKER = "\n\n[Documents indexés"   # note invisible ajoutée au message envoyé à l'agent


def tool_label(name):
    return TOOL_LABELS.get(name, f"🔧 {name}")


def content_to_text(content):
    """Convertit le contenu d'un message en texte (str OU liste de blocs)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


# =========================== Utilitaires ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)   # le plus récent en haut


def reset_chat():
    """Bouton 'New Chat' : nouveau thread + écran vide."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["messages_history"] = []


def load_conversation(thread_id):
    """Récupère les messages d'un thread depuis le checkpointer (SQLite)."""
    state = chat.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def to_display_messages(messages):
    """Garde : messages humains + réponses texte de l'IA.
    Ignore : ToolMessage et AIMessage vides (simples appels d'outils).
    Les outils appelés avant une réponse sont rattachés à cette réponse."""
    display, pending_tools = [], []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = content_to_text(msg.content).split(DOC_MARKER)[0]
            display.append({"role": "user", "content": content, "tools": []})
            pending_tools = []
        elif isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                pending_tools.append(tc["name"])
            text = content_to_text(msg.content)
            if text:
                display.append({"role": "assistant", "content": text, "tools": pending_tools})
                pending_tools = []
    return display


def show_tools_caption(tools):
    if tools:
        st.caption("Outils utilisés : " + ", ".join(tool_label(t) for t in tools))


def get_pending_interrupt(config):
    """[HITL] Interruption en attente sur ce thread ? -> renvoie la question, sinon None."""
    state = chat.get_state(config)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def run_agent(graph_input, config):
    """[HITL] Lance l'agent (nouveau message OU reprise après validation) et affiche la réponse."""
    with st.chat_message("assistant"):
        status = {"box": None}
        used_tools = []

        def ai_only_stream():
            for chunk, meta in chat.stream(graph_input, config=config, stream_mode="messages"):
                if DEBUG:
                    print(type(chunk).__name__, meta.get("langgraph_node"), repr(chunk.content)[:80])

                if isinstance(chunk, AIMessage):   # couvre AIMessage ET AIMessageChunk
                    calls = getattr(chunk, "tool_call_chunks", None) or chunk.tool_calls or []
                    for tc in calls:
                        name = tc.get("name")
                        if name:
                            used_tools.append(name)
                            if status["box"] is None:
                                status["box"] = st.status(f"{tool_label(name)} en cours…", expanded=True)
                            else:
                                status["box"].update(label=f"{tool_label(name)} en cours…", state="running")
                            status["box"].write(f"➡️ Appel de {tool_label(name)}")
                    text = content_to_text(chunk.content)
                    if text:
                        yield text
                elif isinstance(chunk, ToolMessage) and status["box"] is not None:
                    if getattr(chunk, "status", "success") == "error":
                        status["box"].write(f"❌ {tool_label(chunk.name)} a échoué")
                    else:
                        status["box"].write(f"✅ {tool_label(chunk.name)} terminé")

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as e:
            st.error(f"Erreur pendant l'exécution de l'agent : {e}")
            raise

        if not isinstance(ai_message, str):   # write_stream peut renvoyer une liste
            ai_message = "".join(str(x) for x in ai_message) if ai_message else ""

        # Secours : si le stream n'a rien donné, on lit la dernière réponse enregistrée
        if not ai_message:
            msgs = chat.get_state(config).values.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage):
                ai_message = content_to_text(msgs[-1].content)
                if ai_message:
                    st.text(ai_message)

        if status["box"] is not None:
            status["box"].update(label="✅ Outils terminés", state="complete", expanded=False)

        if not ai_message and not used_tools and not get_pending_interrupt(config):
            st.warning("⚠️ L'agent n'a renvoyé aucune réponse. Regarde le terminal (DEBUG) et backend.py.")
            return   # pas de rerun : on laisse l'avertissement visible

    if ai_message or used_tools:
        st.session_state["messages_history"].append(
            {"role": "assistant", "content": ai_message or "", "tools": used_tools}
        )
    st.rerun()   # réaffiche l'historique + les boutons si une validation est en attente


def save_and_ingest(uploaded_file):
    """Écrit le fichier dans UPLOAD_DIR puis appelle ingest_rag_document(path)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    ingest_rag_document(file_path)
    return file_path


# =========================== Session state ===========================
if "messages_history" not in st.session_state:
    st.session_state["messages_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_files" not in st.session_state:   # évite de ré-indexer à chaque rerun
    st.session_state["ingested_files"] = set()
if "doc_names" not in st.session_state:        # noms des PDF indexés (pour prévenir l'agent)
    st.session_state["doc_names"] = []

add_thread(st.session_state["thread_id"])


# =========================== Sidebar ===========================
st.sidebar.title("Agentic Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

if st.session_state["ingested_files"]:
    st.sidebar.caption(f"📄 {len(st.session_state['ingested_files'])} document(s) indexé(s)")

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"]:
    label = f"🟢 {thread_id[:8]}" if thread_id == st.session_state["thread_id"] else thread_id[:8]
    if st.sidebar.button(label, key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["messages_history"] = to_display_messages(load_conversation(thread_id))
        st.rerun()


# =========================== Zone de chat ===========================
st.title("Agentic Chatbot with LangGraph")

config = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": st.session_state["thread_id"]},
    "run_name": "chat_trace",
}
pending = get_pending_interrupt(config)

for message in st.session_state["messages_history"]:
    with st.chat_message(message["role"]):
        if message.get("files"):
            st.caption("📎 " + ", ".join(message["files"]))
        if message["content"]:
            st.text(message["content"])
        show_tools_caption(message.get("tools", []))

# [HITL] Boutons de validation si l'agent attend une décision
if pending:
    with st.chat_message("assistant"):
        st.warning(f"⏸️ Validation requise : {pending}")
        col1, col2 = st.columns(2)
        approve = col1.button("✅ Approuver", use_container_width=True)
        refuse = col2.button("❌ Refuser", use_container_width=True)
    if approve or refuse:
        run_agent(Command(resume="yes" if approve else "no"), config)

# Barre de chat avec bouton "+" pour joindre des PDF
prompt = st.chat_input(
    "Type here",
    accept_file="multiple",
    file_type=["pdf"],
    disabled=bool(pending),   # bloqué tant qu'une validation est en attente
)

user_input = None
file_names = []
if prompt:
    user_input = prompt.text      # texte tapé (peut être vide)
    files = prompt.files or []    # PDF joints (peut être vide)

    # 1) Indexer les PDF joints
    file_names = [f.name for f in files]
    if files:
        with st.chat_message("user"):
            st.caption("📎 " + ", ".join(file_names))
            for uploaded_file in files:
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if file_key in st.session_state["ingested_files"]:
                    continue
                with st.status(f"Indexation de {uploaded_file.name}…", expanded=False) as box:
                    try:
                        save_and_ingest(uploaded_file)
                        st.session_state["ingested_files"].add(file_key)
                        st.session_state["doc_names"].append(uploaded_file.name)
                        box.update(label=f"✅ {uploaded_file.name} indexé", state="complete")
                    except Exception as e:
                        box.update(label=f"❌ Échec : {uploaded_file.name}", state="error")
                        box.write(str(e))

    # Fichiers seuls sans texte : on mémorise et on s'arrête là
    if not user_input and file_names:
        st.session_state["messages_history"].append(
            {"role": "user", "content": "", "tools": [], "files": file_names}
        )

if user_input:
    st.session_state["messages_history"].append(
        {"role": "user", "content": user_input, "tools": [], "files": file_names}
    )
    with st.chat_message("user"):
        st.text(user_input)

    # On signale à l'agent les documents disponibles (non affiché à l'écran)
    agent_text = user_input
    if st.session_state["doc_names"]:
        agent_text += (
            f"{DOC_MARKER} disponibles via l'outil RAG : "
            + ", ".join(st.session_state["doc_names"])
            + ". Utilise cet outil pour toute question sur ces documents.]"
        )
    run_agent({"messages": [HumanMessage(content=agent_text)]}, config)