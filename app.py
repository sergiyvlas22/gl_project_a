import streamlit as st
import requests
import json
import os
import time
import chromadb
from chromadb.config import Settings

OLLAMA_URL = "http://localhost:11434/api/chat"
EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
MODEL = "gemma:2b" 

def embed_text(text: str) -> list[float]:
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=120)
    r.raise_for_status()
    return r.json()["embedding"]

@st.cache_resource
def get_chroma_client():
    return chromadb.Client(
        Settings(
            persist_directory="./chroma",
            is_persistent=True,
        )
    )

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection("chat_memory")
# ЛТМ
ltm = chroma_client.get_or_create_collection(
    name="ltm_memory",
    metadata={"hnsw:space": "cosine"}  # ембедінг норма
)

def should_save_to_ltm(text) -> bool:
    if not text:
        return False
    t = str(text).lower()


    keywords = [
        "remember this", "remember that", "my name is", "i am", "i work",
        "i like", "i prefer", "my favorite", "i usually", "my project",
        "important", "note that", "deadline", "todo", "task",
        "don't forget", "please remember",
    ]
    return any(k in t for k in keywords) or len(t) > 150

# web search(wikipedia)
def tool_search_web(query: str) -> str:
    headers = {
        "User-Agent": "LocalBot/1.0 (your_email@example.com)"
    }

    try:
        # search
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json"
            },
            headers=headers,
            timeout=20
        )
        s.raise_for_status()
        data = s.json()
        results = data["query"]["search"]

        if not results:
            return "No results found on Wikipedia."

        title = results[0]["title"]

        # summary
        summ = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "%20"),
            headers=headers,
            timeout=20
        )
        summ.raise_for_status()
        sd = summ.json()
        extract = sd.get("extract", "")

        return f"Top result: {title}\n\nSummary:\n{extract}"

    except Exception as e:
        return f"Search error: {e}"


def ltm_save(text: str, meta: dict | None = None):
    """Save one memory item into LTM with embeddings."""
    emb = embed_text(text)
    mem_id = f"ltm_{abs(hash(text))}_{len(text)}"  
    ltm.add(
        ids=[mem_id],
        documents=[text],
        embeddings=[emb],
        metadatas=[meta or {"type": "memory"}],
    )
def ltm_retrieve(query: str, k: int = 4) -> str:
    try:
        q_emb = embed_text(query)
        res = ltm.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        if not docs:
            return ""

        out = []
        for d, m in zip(docs, metas):
            tag = (m or {}).get("type", "memory")
            out.append(f"- ({tag}) {d}")
        return "\n".join(out)

    except Exception:
        return ""
    
    #sys prompt
def build_system_prompt(user_profile: dict, persona: dict) -> str:
    user_name = (user_profile.get("name") or "User").strip()
    user_info = (user_profile.get("info") or "").strip()

    persona_name = (persona.get("name") or "Assistant").strip()
    persona_role = (persona.get("role") or "Personal Assistant").strip()
    instructions = (persona.get("instructions") or "").strip()

    parts = [
        f"You are {persona_name}.",
        f"Role: {persona_role}.",
        f"User: {user_name}.",
    ]
    if user_info:
        parts.append(f"User info: {user_info}")
    if instructions:
        parts.append("System instructions:")
        parts.append(instructions)

    return "\n".join(parts)


st.set_page_config(page_title="Persona Chat", layout="wide")
st.title("Persona Chat")

# Сайдбар
with st.sidebar:
    st.header("Settings")
# Юзер профіль
    st.subheader("User Profile")
    up_name = st.text_input("Name", value="Serhii")
    up_info = st.text_area("Basic info", height=80, placeholder="e.g. user, type of answers..")

    st.divider()
# Персона
    st.subheader("Agent Persona")
    p_name = st.text_input("Persona name", value="LocalBot")
    p_role = st.selectbox("Role", ["Personal Assistant", "Grumpy Coder"])
    p_instr = st.text_area("System Instructions", height=120, placeholder="For example, reply with diving into details")

    show_preview = st.checkbox("Show system prompt preview", value=False)
    # debug ltm
    st.subheader("LTM Debug")
    show_ltm = st.checkbox("Show LTM retrieved context", value=True)
    force_save = st.checkbox("Force save every message to LTM", value=False)

# Видалення ретрів лтм (історія)
if "last_ltm_context" not in st.session_state:
    st.session_state.last_ltm_context = ""

if st.button("Clear Retrieved LTM history"):
    st.session_state.last_ltm_context = ""

user_profile = {"name": up_name, "info": up_info}
persona = {"name": p_name, "role": p_role, "instructions": p_instr}
system_prompt = build_system_prompt(user_profile, persona)

#Дані
if st.button("Clear LTM database"):
    chroma_client.delete_collection("ltm_memory")
    ltm = chroma_client.get_or_create_collection(
        name="ltm_memory",
        metadata={"hnsw:space": "cosine"}
    )
    st.session_state.last_ltm_context = ""
    st.sidebar.success("LTM wiped.")
    st.stop()

if show_preview:
    st.code(system_prompt)

# Чат 
if "messages" not in st.session_state:
    st.session_state.messages = []  #short-term

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Write a message..")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

        # Web search
    if prompt.strip().lower().startswith("search:"):
        query = prompt.strip()[7:].strip()
        observation = tool_search_web(query)

        with st.chat_message("assistant"):
            st.markdown(observation)

        st.session_state.messages.append({"role": "assistant", "content": observation})

        # Збереження результату у ЛТМ
        if force_save:
            ltm_save(f"WebSearch: {query}\nResult: {observation}", meta={"type": "web_search"})

        st.stop()

    # LTM request
    ltm_context = ltm_retrieve(prompt, k=4)
    st.session_state.last_ltm_context = ltm_context

    if show_ltm:
        st.sidebar.markdown("### Retrieved LTM")
        st.sidebar.code(ltm_context or "(empty)")

    system_with_memory = system_prompt
    if ltm_context:
        system_with_memory += "\n\nRelevant long-term memory:\n" + ltm_context

    # STM history (ollama)
    messages_for_model = [{"role": "system", "content": system_with_memory}]
    messages_for_model += st.session_state.messages

    payload = {
        "model": MODEL,
        "messages": messages_for_model,
        "stream": False,
    }

    with st.chat_message("assistant"):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            reply = data["message"]["content"]
        except Exception as e:
            reply = f"Error calling Ollama: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    # збереження лтм
    if force_save or should_save_to_ltm(prompt) or should_save_to_ltm(reply):
        ltm_save(f"User: {prompt}\nAssistant: {reply}", meta={"type": "chat_turn"})

