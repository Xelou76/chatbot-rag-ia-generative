"""
TP : Chatbot basé sur un RAG (Retrieval-Augmented Generation)
Code de départ : complétez les fonctions marquées « À COMPLÉTER ».
"""

import os
import re
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Configuration (fournie) -----------------------------------------------------
# Fournisseur : "openai" ou "mistral". Par défaut, détecté selon la clé présente dans .env.
PROVIDER = os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "mistral")
PROVIDER = PROVIDER.strip().lower()
DEFAULT_MODELS = {
    "openai": ("gpt-6-luna", "text-embedding-3-small"),
    "mistral": ("mistral-small-latest", "mistral-embed"),
}
if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER inconnu : '{PROVIDER}' (attendu : openai ou mistral)")
CHAT_MODEL = os.getenv("CHAT_MODEL", DEFAULT_MODELS[PROVIDER][0])
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODELS[PROVIDER][1])
# Une base par fournisseur : des embeddings de modèles différents ne sont pas compatibles
# (chemin relatif au dossier de main.py, pour que la base soit retrouvée quel que soit le dossier de lancement)
PERSIST_DIR = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), f"chroma_db_{PROVIDER}"))

PAGES = [
    "Intelligence_artificielle_générative",
    "Transformeur_génératif_préentraîné",
    "Google_Gemini",
    "Grand_modèle_de_langage",
    "ChatGPT",
    "LLaMA",
    "Réseaux_antagonistes_génératifs",
    "Apprentissage_auto-supervisé",
    "Apprentissage_par_renforcement",
    "DALL-E",
    "Midjourney",
    "Stable_Diffusion"
]


# --- Modèles (fournis) : utilisez get_embeddings() et get_llm() dans votre code ---

def get_embeddings():
    """Renvoie le modèle d'embeddings du fournisseur choisi."""
    if PROVIDER == "mistral":
        from langchain_mistralai import MistralAIEmbeddings
        from tokenizers import Regex, Tokenizer, models, pre_tokenizers
        # Sans tokenizer fourni, langchain-mistralai tente d'en télécharger un sur Hugging Face
        # (dépôt protégé, échoue derrière un proxy). Il ne sert qu'à grouper les textes par lots :
        # on utilise un tokenizer local « 1 caractère = 1 token », qui surestime volontairement.
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
        # max_retries / wait_time : l'offre gratuite limite le débit, on réessaie en cas d'erreur 429
        return MistralAIEmbeddings(model=EMBEDDING_MODEL, tokenizer=tokenizer,
                                   max_retries=10, wait_time=5)
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm():
    """Renvoie le modèle de chat du fournisseur choisi."""
    if PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=CHAT_MODEL, temperature=0, max_retries=5)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=CHAT_MODEL)


def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    URL = "https://fr.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1,  # suit les redirections (sinon une page redirigée revient vide)
    }
    headers = {"User-Agent": "RAG_project/0.0.1"}

    response = requests.get(URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


# --- Paramètres du RAG -----------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def _get_wikipedia_page_with_retry(title, attempts=6, delay=2.0):
    """get_wikipedia_page, en réessayant après une erreur réseau ou un refus temporaire (HTTP 429 de Wikipedia)."""
    for attempt in range(attempts):
        try:
            return get_wikipedia_page(title)
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * 2 ** attempt)


def load_documents(pages=PAGES):
    """
    Partie 1 — Télécharge toutes les pages de PAGES et renvoie une liste de Document
    (texte + métadonnées "title" et "source").
    """
    documents = []
    for title in pages:
        try:
            text = _get_wikipedia_page_with_retry(title)
        except requests.RequestException as error:
            print(f"[page ignorée] {title} : échec du téléchargement ({error})")
            continue
        if not text or not text.strip():
            print(f"[page ignorée] {title} : page vide ou introuvable")
            continue
        documents.append(Document(
            page_content=text,
            metadata={"title": title.replace("_", " "),
                      "source": f"https://fr.wikipedia.org/wiki/{title}"},
        ))
    return documents


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Partie 2 — Découpe les documents en chunks ; les métadonnées sont recopiées dans chaque chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              add_start_index=True)
    return splitter.split_documents(documents)


def get_vectorstore(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, persist_dir=PERSIST_DIR, verbose=True):
    """
    Partie 3 — Base Chroma persistée sur le disque. Si elle contient déjà des chunks, elle est
    rechargée telle quelle ; sinon les pages sont téléchargées, découpées et indexées.
    Une collection par réglage de découpage, pour pouvoir comparer plusieurs tailles (question 30).
    """
    vectorstore = Chroma(
        collection_name=f"wikipedia_ia_{chunk_size}_{chunk_overlap}",
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    n_stored = vectorstore._collection.count()
    if n_stored > 0:
        if verbose:
            print(f"Base rechargée depuis {persist_dir} ({n_stored} chunks)")
        return vectorstore

    documents = load_documents()
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise RuntimeError("Aucun document n'a pu être chargé : vérifiez la connexion à Wikipedia.")
    if verbose:
        print(f"{len(documents)} pages, {len(chunks)} chunks : calcul des embeddings...")
    try:
        vectorstore.add_documents(chunks)
    except Exception:
        # Une indexation interrompue laisserait une base incomplète, qui serait rechargée au lancement suivant
        vectorstore.delete_collection()
        raise
    if verbose:
        print(f"Base créée dans {persist_dir} ({vectorstore._collection.count()} chunks)")
    return vectorstore


def get_retriever(k=TOP_K, search_type="similarity", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                  persist_dir=PERSIST_DIR, verbose=True):
    """
    Parties 2 et 3 — Découpe les documents, calcule les embeddings, les stocke dans
    Chroma (persisté dans PERSIST_DIR) et renvoie un retriever.
    search_type : "similarity" (par défaut) ou "mmr" (question 31).
    """
    vectorstore = get_vectorstore(chunk_size, chunk_overlap, persist_dir, verbose)
    search_kwargs = {"k": k}
    if search_type == "mmr":
        # MMR choisit k chunks parmi les fetch_k plus proches, en pénalisant les chunks redondants
        search_kwargs.update(fetch_k=max(20, 5 * k), lambda_mult=0.5)
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


@lru_cache(maxsize=1)
def _default_retriever():
    return get_retriever(verbose=False)


def format_docs(docs):
    """Partie 4 — Concatène les chunks récupérés en un seul texte de contexte."""
    return "\n\n---\n\n".join(
        f"[{i}] Source : {doc.metadata.get('title', 'inconnue')}\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs, start=1)
    )


def build_prompt(only_context=True, say_unknown=True, citations=False):
    """
    Partie 4 — Prompt du RAG, avec deux variables : {context} et {question}.
    only_context / say_unknown permettent de retirer les consignes (question 17) ;
    citations demande de citer le numéro des passages (question 33).
    """
    rules = ["Réponds toujours en français, de façon claire et concise."]
    if only_context:
        rules.append("Réponds uniquement à partir du contexte fourni ci-dessous, sans utiliser tes propres connaissances.")
    else:
        rules.append("Tu peux t'appuyer sur le contexte fourni ci-dessous.")
    if say_unknown:
        rules.append("Si la réponse ne figure pas dans le contexte, dis simplement que tu ne sais pas.")
    if citations:
        rules.append("Après chaque affirmation, indique entre crochets le numéro du ou des passages utilisés, "
                     "par exemple [1] ou [2][3].")
    system = ("Tu es un assistant qui répond à des questions sur l'intelligence artificielle générative.\n"
              + "\n".join(f"- {rule}" for rule in rules)
              + "\n\nContexte (passages numérotés) :\n{context}")
    return ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])


PROMPT = build_prompt()


# --- Bonus : reformulations (question 29) et mémoire conversationnelle (question 32) ---

def generate_queries(query, n=3):
    """Demande au LLM n reformulations de la question, une par ligne."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu aides un moteur de recherche documentaire sur l'IA générative. Donne exactement {n} "
                   "reformulations différentes de la question de l'utilisateur, en français, une par ligne, "
                   "sans numérotation ni commentaire."),
        ("human", "{question}"),
    ])
    text = (prompt | get_llm() | StrOutputParser()).invoke({"n": n, "question": query})
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line][:n]


def reciprocal_rank_fusion(rankings, c=60):
    """Fusionne plusieurs listes de Document : score = somme de 1 / (c + rang)."""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = (doc.metadata.get("source"), doc.metadata.get("start_index"), doc.page_content[:50])
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1 / (c + rank)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


def multi_query_retrieve(query, retriever, n=3):
    """RAG-Fusion : recherche avec la question et n reformulations, puis fusion par RRF."""
    queries = [query] + generate_queries(query, n)
    rankings = retriever.batch(queries)
    return reciprocal_rank_fusion(rankings)[:len(rankings[0])]


def contextualize_question(query, history):
    """Reformule une question de suivi (« Et qui l'a créé ? ») en question autonome, à l'aide de l'historique."""
    if not history:
        return query
    transcript = "\n".join(f"{'Utilisateur' if role == 'user' else 'Assistant'} : {content}"
                           for role, content in history[-6:])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "À partir de l'historique de conversation et de la dernière question de l'utilisateur, "
                   "réécris cette question pour qu'elle soit compréhensible seule (remplace les pronoms par ce "
                   "qu'ils désignent). Ne réponds pas à la question. Si elle est déjà autonome, renvoie-la telle "
                   "quelle.\n\nHistorique :\n{history}"),
        ("human", "{question}"),
    ])
    return (prompt | get_llm() | StrOutputParser()).invoke({"history": transcript, "question": query}).strip()


def rag_answer(query, retriever=None, stream=True, prompt=None, history=None, multi_query=False):
    """
    Partie 4 — Récupère les chunks pertinents, interroge le LLM et affiche
    la réponse puis ses sources.
    stream=False : pas d'affichage dans le terminal (interface Streamlit).
    history : liste de (rôle, texte) pour la mémoire conversationnelle (question 32).
    Renvoie (réponse, liste des chunks utilisés).
    """
    retriever = retriever or _default_retriever()
    question = contextualize_question(query, history)
    docs = multi_query_retrieve(question, retriever) if multi_query else retriever.invoke(question)

    chain = (prompt or PROMPT) | get_llm() | StrOutputParser()
    inputs = {"context": format_docs(docs), "question": question}
    sources = list(dict.fromkeys(doc.metadata["source"] for doc in docs))

    if not stream:
        return chain.invoke(inputs), docs

    if question != query:
        print(f"(question reformulée : {question})\n")
    parts = []
    for token in chain.stream(inputs):
        print(token, end="", flush=True)
        parts.append(token)
    print("\n\nSources :")
    for source in sources:
        print(f"  - {source}")
    return "".join(parts), docs


if __name__ == "__main__":
    print(f'Fournisseur : {PROVIDER} (chat : {CHAT_MODEL}, embeddings : {EMBEDDING_MODEL})')
    print('Building retriever...')
    retriever = get_retriever()

    try:
        while True:
            print('-' * 50)
            print('Posez une question :')
            question = input('> ').strip()
            if not question:
                continue
            print()
            rag_answer(question, retriever)
            print('\n')

    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
