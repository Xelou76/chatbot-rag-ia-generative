"""
TP : Chatbot RAG — interface web Streamlit (parties 6 et 8).
Lancement : streamlit run app.py
"""

import hmac
import os

import streamlit as st

# Question 25 — En ligne, il n'y a pas de .env : les clés sont dans st.secrets. On les copie dans
# os.environ AVANT d'importer main.py, qui choisit le fournisseur au moment de l'import.
# En local, sans .streamlit/secrets.toml, l'accès à st.secrets lève une exception : on l'ignore.
for key in ("OPENAI_API_KEY", "MISTRAL_API_KEY", "LLM_PROVIDER", "APP_PASSWORD"):
    try:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

from main import CHAT_MODEL, EMBEDDING_MODEL, PROVIDER, TOP_K, build_prompt, get_retriever, rag_answer  # noqa: E402

st.set_page_config(page_title="Chatbot RAG — IA générative", page_icon="🤖")
st.title("🤖 Chatbot RAG sur l'IA générative")
st.caption(f"Réponses fondées sur 12 pages Wikipedia · {PROVIDER} ({CHAT_MODEL}, {EMBEDDING_MODEL})")

# Question 27 — L'application publique consomme la clé API : si APP_PASSWORD est défini (secret ou
# variable d'environnement), on le demande avant tout appel au fournisseur. Sans APP_PASSWORD, accès libre.
APP_PASSWORD = os.environ.get("APP_PASSWORD")
if APP_PASSWORD and not st.session_state.get("authenticated"):
    password = st.text_input("Mot de passe", type="password")
    if password and hmac.compare_digest(password.encode(), APP_PASSWORD.encode()):
        st.session_state.authenticated = True
        st.rerun()
    if password:
        st.error("Mot de passe incorrect.")
    st.stop()


@st.cache_resource(show_spinner="Construction de la base vectorielle (une seule fois)...")
def load_retriever(search_type):
    # Exécuté une seule fois par valeur de search_type, puis gardé en mémoire entre les interactions
    return get_retriever(k=TOP_K, search_type=search_type, verbose=False)


with st.sidebar:
    st.header("Options")
    search_type = "mmr" if st.toggle("Recherche MMR (résultats plus variés)", value=False) else "similarity"
    multi_query = st.toggle("Reformulation de la question (RAG-Fusion)", value=False)
    use_memory = st.toggle("Mémoire conversationnelle", value=True,
                           help="Reformule les questions de suivi (« Et qui l'a créé ? ») à l'aide de l'historique.")
    citations = st.toggle("Citer les passages [1], [2]...", value=True)
    if st.button("Effacer la conversation"):
        st.session_state.messages = []

retriever = load_retriever(search_type)

if "messages" not in st.session_state:
    st.session_state.messages = []  # chaque message : {"role", "content", "sources", "passages"}


def show_sources_and_passages(message):
    if message.get("sources"):
        st.markdown("**Sources :** " + " · ".join(f"[{title}]({url})" for title, url in message["sources"]))
    if message.get("passages"):
        # Question 20 — passages récupérés, dans une zone repliable
        with st.expander(f"Voir les {len(message['passages'])} passages récupérés"):
            for i, (title, text) in enumerate(message["passages"], start=1):
                st.markdown(f"**[{i}] {title}**")
                st.text(text)


# Historique de la conversation (Streamlit ré-exécute le script à chaque interaction)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources_and_passages(message)

if question := st.chat_input("Posez une question sur l'IA générative..."):
    history = [(m["role"], m["content"]) for m in st.session_state.messages] if use_memory else None
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des passages et rédaction de la réponse..."):
            try:
                answer, docs = rag_answer(question, retriever, stream=False, history=history,
                                          multi_query=multi_query, prompt=build_prompt(citations=citations))
            except Exception as error:  # clé invalide, quota dépassé, réseau...
                answer, docs = f"Désolé, une erreur est survenue : `{error}`", []
        sources = list(dict.fromkeys((d.metadata["title"], d.metadata["source"]) for d in docs))
        message = {"role": "assistant", "content": answer, "sources": sources,
                   "passages": [(d.metadata["title"], d.page_content) for d in docs]}
        st.markdown(answer)
        show_sources_and_passages(message)
    st.session_state.messages.append(message)
