# Chatbot RAG sur l'IA générative

Chatbot de type **RAG** (*Retrieval-Augmented Generation*) qui répond à des questions sur l'IA générative
à partir de **12 pages Wikipedia en français** (ChatGPT, LLaMA, Stable Diffusion, GAN, etc.).
Chaque réponse est fondée sur les passages retrouvés dans ces pages et affiche ses sources.

Pipeline : pages Wikipedia → découpage en chunks (1 000 caractères, chevauchement de 200) → embeddings →
base vectorielle Chroma → recherche des 4 passages les plus proches → réponse du LLM (LangChain).

## Prérequis

- Python 3.10 ou plus (testé avec 3.12 et 3.13)
- Une clé API **OpenAI** ou **Mistral AI** (gratuite : <https://console.mistral.ai>, offre *Experiment*)

## Installation

```bash
git clone https://github.com/<votre-compte>/<votre-depot>.git
cd <votre-depot>
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.template .env              # Windows : copy .env.template .env
```

Renseignez ensuite **une** des deux clés dans `.env` :

```
OPENAI_API_KEY=sk-...     # modèles gpt-6-luna + text-embedding-3-small
MISTRAL_API_KEY=...       # modèles mistral-small-latest + mistral-embed
```

Le fournisseur est choisi automatiquement (OpenAI si sa clé est présente) ; on peut le forcer avec `LLM_PROVIDER`.
Le fichier `.env` contient un secret : il est exclu de Git par le `.gitignore`.

## Utilisation

**En ligne de commande :**

```bash
python main.py
```

Au premier lancement, les pages sont téléchargées et indexées dans `chroma_db_<fournisseur>/` (quelques minutes) ;
les lancements suivants rechargent la base depuis le disque.

**Interface web (Streamlit) :**

```bash
streamlit run app.py
```

puis ouvrez <http://localhost:8501>. La barre latérale permet d'activer la recherche MMR, la reformulation
de la question (RAG-Fusion), la mémoire conversationnelle et les citations `[1]`, `[2]`.

**Avec Docker :**

```bash
docker build -t tp-rag .
docker run -p 8501:8501 -e MISTRAL_API_KEY=... tp-rag
```

## Déploiement sur Streamlit Community Cloud

1. Sur <https://share.streamlit.io>, *Create app* → *Yup, I have an app* → ce dépôt, branche `main`, fichier `app.py`.
2. Dans *Advanced settings* : Python 3.12, et le secret au format TOML : `MISTRAL_API_KEY = "votre-cle"`.

`app.py` copie ce secret dans les variables d'environnement avant de charger `main.py`.

## Structure du projet

| Fichier | Rôle |
|---|---|
| `main.py` | pipeline RAG : chargement, découpage, base Chroma, prompt, `rag_answer` (CLI) |
| `app.py` | interface web Streamlit |
| `experiences.py` | expériences du compte rendu (`python experiences.py partie5 q30 ...`) |
| `resultats/` | sorties des expériences |
| `COMPTE_RENDU.md` | réponses aux questions du TP |
| `Dockerfile`, `.dockerignore` | image Docker de l'application |
