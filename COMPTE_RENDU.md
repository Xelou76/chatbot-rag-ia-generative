# TP — Chatbot RAG : compte rendu

Fournisseur utilisé : **OpenAI** (chat `gpt-6-luna`, embeddings `text-embedding-3-small`), Python 3.13, LangChain 1.4, Chroma.
Les questions de réflexion demandées dans le compte rendu sont marquées **[CR]**. Toutes les mesures citées sont
reproductibles avec `python experiences.py <nom>` et leurs sorties brutes sont dans le dossier `resultats/`.

---

## Contexte

### Question 1 [CR] — Pourquoi ne pas donner les 12 pages entières au LLM ?

- **Coût et latence** : les 12 pages font 274 520 caractères, soit environ **74 800 tokens**. Les envoyer à chaque question coûterait environ **96 fois plus** de tokens qu'un prompt RAG (776 tokens en moyenne, question 34), et serait d'autant plus lent.
- **Taille de la fenêtre de contexte** : un corpus réel (des milliers de pages, une documentation d'entreprise) dépasse de loin la fenêtre de n'importe quel modèle. Le RAG passe à l'échelle, pas le « tout dans le prompt ».
- **Qualité** : noyer l'information utile dans 75 000 tokens de texte sans rapport la dilue. Les modèles exploitent moins bien l'information placée au milieu d'un long contexte (*lost in the middle*), et les passages hors sujet sont autant de distracteurs.
- **Traçabilité** : avec quelques passages récupérés, on peut afficher précisément les sources de la réponse et les vérifier.

---

## Mise en place

**Question 2.** Environnement virtuel `.venv` créé et dépendances installées (`pip install -r requirements.txt`, plus `tiktoken` et `rank_bm25` pour les bonus).

**Question 3.** Fichier `.env` créé à partir de `.env.template`, avec `OPENAI_API_KEY`. Il est exclu de Git (question 21).

**Question 4.** Au démarrage, le programme affiche :
`Fournisseur : openai (chat : gpt-6-luna, embeddings : text-embedding-3-small)`, puis `Building retriever...`, et s'arrête sur `NotImplementedError` dans `get_retriever()`, comme prévu.

---

## Partie 1 — Collecte des documents

**Question 5.** La page « ChatGPT » fait **55 309 caractères**. Le texte est brut (`explaintext`) : les titres de sections apparaissent avec la syntaxe wiki, entourés de signes `=` dont le nombre donne le niveau, et précédés de lignes vides : `== Caractéristiques ==`, `=== Modèles ===`, `==== ChatGPT Atlas, le navigateur web d'OpenAI propulsé par l'IA ====`. On remarque aussi que les formules mathématiques (page GAN) sont rendues en LaTeX éclaté (`{\displaystyle ...}`), ce qui pollue certains chunks.

**Question 6.** `load_documents()` renvoie un `Document` par page, avec `metadata = {"title": titre sans « _ », "source": "https://fr.wikipedia.org/wiki/<titre>"}`. Les pages vides et les erreurs réseau (`requests.RequestException`) sont signalées et ignorées. En pratique, la connexion était instable et Wikipedia renvoie des erreurs **429** (trop de requêtes) quand on enchaîne les appels : j'ai ajouté une fonction `_get_wikipedia_page_with_retry` qui réessaie jusqu'à 6 fois avec un délai croissant (2, 4, 8… s). Sans elle, 2 pages sur 12 (« Apprentissage auto-supervisé », « DALL-E ») étaient perdues lors d'un essai. Résultat final : **12 pages**, de 3 810 caractères (Transformeur génératif préentraîné) à 55 309 (ChatGPT).

---

## Partie 2 — Découpage en chunks

**Question 7.** `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` donne **425 chunks** au total. Les métadonnées `title` et `source` sont bien recopiées dans chaque chunk ; j'ai ajouté `add_start_index=True`, qui ajoute la position du chunk dans la page (`start_index`), utile pour les retrouver et dédoublonner.

### Question 8 [CR] — Chevauchement, chunks trop petits, trop grands

- **Le chevauchement** évite qu'une information soit coupée en deux à la frontière entre deux chunks : une phrase ou une idée à cheval sur la frontière se retrouve entière dans au moins un des deux, et chaque chunk garde un peu du contexte qui le précède.
- **Chunks trop petits** : un chunk perd son contexte. Il ne contient plus le sujet dont il parle (« il a été entraîné sur… » sans dire quel modèle), son embedding représente mal la question, et une réponse qui demande deux phrases voisines est répartie sur plusieurs chunks qu'il faut tous retrouver. Il y a aussi beaucoup plus de vecteurs à calculer et à stocker (1 401 chunks à 300 caractères contre 425 à 1 000).
- **Chunks trop grands** : un chunk mélange plusieurs sujets, son embedding (un seul vecteur) devient une moyenne floue et la recherche perd en précision ; chaque chunk consomme beaucoup du budget de contexte (k × taille), ce qui coûte plus cher, laisse moins de place à d'autres sources et ajoute du texte hors sujet qui peut distraire le modèle. Enfin, le modèle d'embeddings a une longueur maximale d'entrée (8 191 tokens pour `text-embedding-3-small`).

---

## Partie 3 — Embeddings et base vectorielle

**Question 9.** `get_embeddings().embed_query("Qu'est-ce qu'un GAN ?")` renvoie un vecteur de **dimension 1 536** (`text-embedding-3-small`). Avec Mistral (`mistral-embed`), la dimension est de 1 024.

**[CR]** On ne peut pas mélanger dans une même base des vecteurs de deux modèles : chaque modèle apprend son **propre espace vectoriel**. Les dimensions ne correspondent même pas (1 536 contre 1 024, Chroma refuserait l'insertion), et même à dimension égale, la coordonnée n°17 d'un modèle n'a aucun rapport avec celle de l'autre : une similarité cosinus entre une question encodée par l'un et un passage encodé par l'autre ne mesure rien. La question et les documents doivent être encodés par le même modèle, d'où une base par fournisseur (`chroma_db_openai`, `chroma_db_mistral`).

### Question 10 [CR] — BM25 contre embeddings

J'ai comparé les deux méthodes sur les 425 chunks (`python experiences.py q10`) : rang du premier chunk qui contient l'information.

| Question | Rang BM25 | Rang embeddings |
|---|---|---|
| « Quel logiciel d'OpenAI dessine à partir d'une phrase ? » | absent du top 20 | **5** |
| « Que fait Emad Mostaque ? » | **1** | absent du top 20 |
| « Que dit William Orrick ? » | **1** | 5 |
| « Qui est Greg Brockman ? » | **1** | 4 |

- **BM25 échoue, les embeddings réussissent** : « Quel logiciel d'OpenAI dessine à partir d'une phrase ? ». La page parle d'un programme qui « produit des images à partir de descriptions textuelles » : aucun mot en commun (*dessine*, *phrase*), mais le même sens. Les embeddings retrouvent DALL-E, BM25 non. Même chose pour les synonymes ou une question posée dans une autre langue.
- **Les embeddings échouent, BM25 réussit** : « Que fait Emad Mostaque ? ». Ce nom propre rare n'apparaît qu'une fois dans le corpus ; BM25 lui donne un poids énorme (IDF) et trouve le bon passage en premier, alors que pour l'embedding ce nom inconnu n'est qu'un « nom de personne » parmi d'autres. C'est le cas typique des identifiants exacts : noms propres rares, références de produits, codes d'erreur, numéros d'articles.

**Question 11.** `get_retriever()` charge et découpe les documents, crée une base `Chroma` (collection `wikipedia_ia_1000_200`, distance cosinus) avec `get_embeddings()` et `PERSIST_DIR`, y ajoute les chunks et renvoie `vectorstore.as_retriever(search_kwargs={"k": 4})`. Test avec « Qui a créé LLaMA ? » : les 3 premiers chunks viennent de la page LLaMA, et le premier contient la réponse (« une famille de grands modèles de langage développé par **Meta** »). Le 4ᵉ (sur la part des données francophones dans l'entraînement des LLM) est hors sujet : le retriever renvoie toujours k chunks, même quand moins sont pertinents. L'indexation a pris **253 s**, ralentie par le réseau.

**Question 12.** Après la création de l'objet `Chroma` avec le même `persist_directory`, si `vectorstore._collection.count() > 0`, la base est rechargée sans rien recalculer : **0,0 s** au lieu de 253 s. Deux précautions supplémentaires :
- si l'indexation échoue en cours de route, la collection est supprimée, pour ne pas recharger plus tard une base incomplète ;
- le nom de la collection contient les paramètres de découpage, pour que changer `chunk_size` crée une nouvelle collection au lieu de réutiliser l'ancienne (question 30).

Limite : si l'on change la liste des pages, il faut supprimer le dossier `chroma_db_*` pour reconstruire la base.

---

## Partie 4 — Génération de la réponse

**Question 13.** `format_docs` numérote les chunks et indique la page d'origine, séparés par `---` :
```
[1] Source : LLaMA
LLaMA (Large Language Model Meta AI) est une famille ...
---
[2] Source : LLaMA
...
```

**Question 14.** Le `ChatPromptTemplate` contient un message *system* (répondre en français, **uniquement** à partir du contexte, dire « je ne sais pas » si la réponse n'y figure pas, puis le contexte `{context}`) et un message *human* `{question}`. Il est construit par `build_prompt()`, dont les options servent aux questions 17 et 33.

**Question 15.** `rag_answer(query)` récupère les chunks (`retriever.invoke`), construit `prompt | get_llm() | StrOutputParser()`, affiche la réponse au fil de l'eau avec `chain.stream(...)`, puis les sources dédoublonnées avec `list(dict.fromkeys(...))`, et renvoie `(réponse, docs)`. Le paramètre `stream=False` désactive l'affichage pour Streamlit.

---

## Partie 5 — Tests du chatbot

### Question 16 — Réponses et sources

| Question | Réponse du chatbot | Sources |
|---|---|---|
| (i) Qui a développé Stable Diffusion ? | « Le contexte fourni ne précise pas qui a développé Stable Diffusion. » | Stable Diffusion |
| (ii) Différence entre un GAN et un modèle de diffusion ? | Un GAN met en compétition un générateur et un discriminateur ; un modèle de diffusion débruite progressivement une image pour faire apparaître le motif demandé. | IA générative, Réseaux antagonistes génératifs |
| (iii) En quelle année ChatGPT a-t-il été lancé ? | « ChatGPT a été lancé en 2022. » (parfois « en novembre 2022 ») | ChatGPT |
| (iv) Quelle est la capitale de l'Australie ? | « Je ne sais pas. » | Apprentissage par renforcement, IA générative, Midjourney |

(ii) et (iii) sont correctes. (i) est un échec, analysé à la question 18. Pour (iv), les 4 chunks récupérés n'ont aucun rapport avec la question.

### Question 17 [CR] — Question hors corpus

Sur (iv), le chatbot répond **« Je ne sais pas. »** : les 4 chunks récupérés sont hors sujet (le retriever renvoie toujours les k plus proches, même très éloignés) et le prompt interdit d'utiliser autre chose que le contexte.

| Prompt | Réponse à (iv) |
|---|---|
| complet | « Je ne sais pas. » |
| sans « réponds uniquement à partir du contexte » | « Je ne sais pas. » |
| sans cette consigne **ni** « dis que tu ne sais pas » | « La capitale de l'Australie est **Canberra**. » |

Retirer seulement la consigne « uniquement » ne change rien ici, car il reste « si la réponse ne figure pas dans le contexte, dis que tu ne sais pas ». Quand on retire les deux, le modèle répond avec ses **connaissances internes** : la réponse est juste, mais elle n'est plus fondée sur les documents, et les « sources » affichées (Midjourney, apprentissage par renforcement) ne la justifient pas du tout.

**Conclusion** : c'est la consigne qui fait du chatbot un système **fondé sur ses sources**. Sans elle, il mélange contexte et mémoire paramétrique ; c'est inoffensif pour une capitale, mais sur une question pointue, le modèle peut inventer une réponse plausible, présentée avec des sources qui semblent la garantir. La contrepartie de cette rigueur : le modèle refuse parfois de répondre alors que l'information est presque là (question (i)).

### Question 18 [CR] — Questions mal traitées : recherche ou génération ?

1. **« Combien de paramètres compte le plus grand modèle LLaMA 2 ? »** → « Je ne sais pas ». La réponse est dans le corpus (page LLaMA : « Llama 2 … en trois tailles différentes cette fois : 7B, 13B et **70B** »), mais ce chunk n'est **pas parmi les 4 récupérés** : la recherche ramène l'introduction de « Grand modèle de langage » et d'autres chunks de LLaMA. **Problème de recherche.**
2. **(i) « Qui a développé Stable Diffusion ? »** Les 4 chunks viennent bien de la page Stable Diffusion, mais aucun ne dit explicitement qui l'a développé. Ils contiennent seulement « En février 2024, **Stability** publie un modèle Stable Cascade… » et « Articles connexes : **Stability AI** ». La seule phrase explicite (« … contre **Stability AI**, Midjourney et DeviantArt, **créateurs** des générateurs d'art Stable Diffusion, Midjourney et DreamUp ») est dans la section « Poursuites en justice », qui n'est même pas dans les 15 premiers résultats. C'est d'abord un problème de **recherche** : le passage décisif parle de procès et ressemble peu à la question. C'est aussi un choix de la **génération** : avec la consigne stricte, le modèle refuse de déduire « Stability a développé Stable Diffusion » de « Stability publie un modèle Stable Cascade ». Le corpus lui-même est en cause : la page française ne dit nulle part « développé par Stability AI ».
3. **« Lequel est sorti en premier : Midjourney ou Stable Diffusion ? »** → « Je ne sais pas ». Le contexte ne contient que la page Midjourney (bêta le 12 juillet 2022). L'information sur Stable Diffusion est dans une autre page (« … suivie des sorties de Midjourney et Stable Diffusion en 2022 »). Une question **multi-documents** demande plusieurs recherches : **problème de recherche**.

Dans les trois cas, le LLM a raisonné correctement sur ce qu'on lui a donné : le maillon faible est la recherche. Les pistes testées en bonus (reformulation, MMR, k plus grand) et d'autres (recherche hybride BM25 + embeddings, re-ranking, ajout du titre de la page dans chaque chunk) visent ce maillon.

---

## Partie 6 — Interface Streamlit

**Question 19.** `app.py` importe `get_retriever` et `rag_answer` depuis `main.py`. Le retriever est construit une seule fois grâce à `@st.cache_resource`. L'historique est conservé dans `st.session_state.messages` et réaffiché à chaque exécution du script. La question est saisie dans `st.chat_input`, la réponse s'affiche dans `st.chat_message("assistant")`, suivie de liens cliquables vers les pages sources (`[titre](url)`). Les erreurs (clé invalide, quota, réseau) sont affichées dans la conversation au lieu de faire planter l'application. Une barre latérale permet d'activer les bonus : MMR, reformulation de la question, mémoire conversationnelle, citations.

**Question 20.** Chaque réponse est suivie d'un `st.expander("Voir les 4 passages récupérés")` qui affiche les chunks numérotés avec leur page d'origine. L'interface a été testée en local (`streamlit run app.py`) : réponse, sources cliquables, zone repliable, et mémoire (« Qu'est-ce que LLaMA ? » puis « Et qui l'a créé ? » → « LLaMA a été développé par Meta. [1] »).

---

## Partie 7 — Git

### Question 21 [CR] — `.gitignore`

| Exclusion | Justification |
|---|---|
| `.env` (et `.streamlit/secrets.toml`) | Contient la clé API, un **secret** : un dépôt public l'exposerait à tous (consommation des crédits, abus), et une clé poussée reste dans l'historique même après suppression. |
| `chroma_db_*/` | Base vectorielle **régénérable** à partir des pages Wikipedia, volumineuse et binaire (fichiers SQLite et index HNSW) ; elle dépend aussi du fournisseur d'embeddings choisi. |
| `.venv/` | Environnement **propre à chaque machine** (chemins absolus, binaires compilés pour Windows ou Linux), très lourd ; il se recrée avec `pip install -r requirements.txt`. |
| `__pycache__/` | Bytecode Python **généré automatiquement** à l'exécution, spécifique à la version de Python : aucune information utile. |

**Questions 22 et 23.** Le dépôt local est initialisé et le premier commit est fait ; `git status` confirme que `.env` n'est pas suivi. Le `README.md` explique l'installation, la configuration de la clé, le lancement en ligne de commande, Streamlit et Docker, et le déploiement. Le dépôt public est en ligne : <https://github.com/Xelou76/chatbot-rag-ia-generative>, créé avec `gh repo create ... --public --source . --push`. On a vérifié sur GitHub qu'il ne contient ni `.env` ni l'énoncé, et l'historique ne contient aucune clé.

---

## Partie 8 — Déploiement

**Questions 24 à 26.** Déploiement sur Streamlit Community Cloud avec la clé OpenAI dans les secrets (`OPENAI_API_KEY`). L'énoncé recommande une clé personnelle pour une application publique ; comme la clé utilisée est partagée, l'application est protégée par un mot de passe (`APP_PASSWORD`, voir la question 27). Le code est prêt :

**Question 25.** Au début d'`app.py`, avant `from main import ...`, les secrets `OPENAI_API_KEY`, `MISTRAL_API_KEY` et `LLM_PROVIDER` sont copiés de `st.secrets` vers `os.environ` s'ils existent. L'accès est entouré d'un `try / except`, car sans fichier `.streamlit/secrets.toml` (en local), `st.secrets` lève une exception. En local, l'application continue donc d'utiliser le `.env` (vérifié). La copie doit se faire **avant** l'import, car `main.py` choisit le fournisseur au moment de son import.

**Question 26.** *(Comportement attendu, à vérifier une fois l'application déployée.)* Après un `git push`, Streamlit Community Cloud détecte le nouveau commit sur la branche `main` et redéploie automatiquement l'application, qui affiche la nouvelle phrase au bout de quelques secondes : le dépôt GitHub sert de source unique, ce qui fait du `git push` une mise en production (déploiement continu).

### Question 27 [CR] — Risques d'une application publique

- **Coûts** : toute personne qui a le lien consomme les crédits de la clé ; un robot qui enchaîne les requêtes peut vider un quota ou générer une facture importante.
- **Abus** : utilisation du chatbot comme accès gratuit à un LLM, tentatives de *prompt injection* ou de contournement des consignes, attaque par déni de service (requêtes très longues ou très nombreuses), fuite de la clé si elle était écrite dans le code.

**Mesures** :
1. **Limiter le débit** : nombre maximal de questions par session ou par adresse IP, longueur maximale de la question, `max_tokens` sur la réponse.
2. **Restreindre l'accès** : application privée sur Streamlit Cloud (liste d'invités) ou authentification (`st.login`, mot de passe). *Mis en place* : si le secret `APP_PASSWORD` est défini, `app.py` demande ce mot de passe (comparaison à temps constant avec `hmac.compare_digest`) et appelle `st.stop()` tant qu'il n'est pas correct, donc aucune requête n'atteint l'API.
3. **Plafonner les dépenses** chez le fournisseur (budget mensuel, alertes), et utiliser une clé dédiée au déploiement, révocable, à quota limité (par exemple une clé Mistral gratuite).
4. **Surveiller** l'usage (journaux, LangSmith) et mettre en cache les réponses aux questions fréquentes.

### Question 28 (bonus) — Docker

Le `Dockerfile` part de `python:3.12-slim`, installe les dépendances (couche mise en cache tant que `requirements.txt` ne change pas), copie `main.py` et `app.py`, expose le port 8501 et lance `streamlit run app.py --server.address=0.0.0.0`. La clé est passée au lancement, sans jamais entrer dans l'image : `docker run -p 8501:8501 --env-file .env -v tp-rag-chroma:/app/chroma_db_openai tp-rag`. Le volume `tp-rag-chroma` conserve la base Chroma entre deux lancements : sans lui, le conteneur repart d'un système de fichiers vierge et réindexe les 12 pages (environ 4 minutes) à chaque démarrage.

**Test réel (Docker Desktop 4.97, moteur 29.9.2 sous WSL2).** `docker build -t tp-rag .` produit une image de 1,5 Go ; `ls -la /app` dans l'image montre seulement `app.py`, `main.py` et `requirements.txt`, donc **pas de `.env`**. Au premier lancement, le conteneur indexe les 425 chunks dans le volume, puis répond à « Qui a créé LLaMA ? » : « LLaMA a été développé par Meta. [1] », avec les sources LLaMA et Intelligence artificielle générative. Après suppression et relance du conteneur, le volume contient toujours les 425 chunks et l'interface est prête en quelques secondes, sans réindexation. L'option `--browser.gatherUsageStats=false` désactive l'envoi de statistiques d'usage par Streamlit.

**Pourquoi un `.dockerignore` ?** `docker build` envoie tout le dossier au moteur Docker (le *contexte de build*). Sans `.dockerignore`, un `COPY . .` copierait le fichier **`.env` dans l'image** : la clé resterait dans une couche de l'image, lisible par toute personne qui la récupère (par exemple sur Docker Hub). On copierait aussi `.venv` (des centaines de Mo de binaires Windows, inutilisables sous Linux), les bases `chroma_db_*` et `.git`, ce qui alourdit l'image et ralentit le build.

---

## Pour aller plus loin (bonus)

### Question 29 — Multi-query / RAG-Fusion

`multi_query_retrieve` demande au LLM 3 reformulations, lance les 4 recherches (question d'origine comprise) et fusionne les listes par **Reciprocal Rank Fusion** ($\sum 1/(60 + \text{rang})$). Sur les 5 questions de référence et 2 autres :
- les reformulations sont pertinentes (« Quel chercheur a inventé les GAN ? », « Par qui Stable Diffusion a-t-il été développé ? ») ;
- **aucune réponse n'est améliorée** : 4 réponses correctes sur 5 dans les deux cas, et les pages récupérées sont presque identiques ;
- le coût augmente : un appel au LLM et 3 recherches de plus, et une latence médiane qui passe d'environ 1,9 s à 3,7 s (jusqu'à 12 s une fois).

Ici, les reformulations restent sémantiquement très proches de la question d'origine, donc leurs embeddings aussi. La méthode n'a pas corrigé l'échec sur Stable Diffusion, dont le passage décisif parle d'un procès et non de développement. Elle est plus utile pour des questions vagues, mal formulées ou à plusieurs facettes.

### Question 30 — Influence des paramètres

5 questions de référence (Stable Diffusion, ChatGPT, GAN, Midjourney, Gemini) ; « réponse correcte » = le mot-clé attendu figure dans la réponse ; chevauchement = 20 % de la taille.

| chunk_size | k | Réponses correctes | Tokens de contexte (moy.) | Temps par question |
|---|---|---|---|---|
| 300 (1 401 chunks) | 2 | 4/5 | 128 | 3,2 s |
| 300 | 4 | 5/5 | 254 | 1,6 s |
| 300 | 8 | 5/5 | 503 | 1,4 s |
| 1000 (425 chunks) | 2 | 5/5 | 325 | 2,1 s |
| 1000 | 4 | 4/5 | 689 | 1,7 s |
| 1000 | 8 | 5/5 | 1 402 | 2,2 s |
| 2000 (207 chunks) | 2 | 4/5 | 687 | 1,9 s |
| 2000 | 4 | 5/5 | 1 408 | 2,0 s |
| 2000 | 8 | 5/5 | 2 872 | 2,1 s |

**Observations.** Toutes les configurations obtiennent 4 ou 5 sur 5 : les questions factuelles simples sont bien traitées quel que soit le réglage, et **seule la question sur Stable Diffusion varie**. Elle échoue ou réussit d'une exécution à l'autre, y compris pour une même configuration (1000 / k = 4 : 4/5 ici, alors que la question (i) réussit parfois), car le modèle n'est pas déterministe. Avec 5 questions, un écart d'un point n'est pas significatif. Ce qui varie nettement, c'est le **coût** : les tokens de contexte sont proportionnels à `chunk_size × k` (de 128 à 2 872), soit un facteur 22 pour le même résultat. Le réglage par défaut (1 000 / 4) est un bon compromis. Pour départager les tailles, il faudrait davantage de questions et des questions plus difficiles (multi-documents, détails précis), comme celles de la question 18.

### Question 31 — MMR

Avec `search_type="mmr"` (`fetch_k=20`, `lambda_mult=0.5`), le retriever choisit des chunks à la fois proches de la question et **différents entre eux** :
- « Qu'est-ce qu'un GAN ? » : en similarité, 4 chunks de la page GAN ; en MMR, 2 chunks GAN, puis un chunk d'apprentissage par renforcement et un de ChatGPT ;
- « Qui a créé LLaMA ? » : le 3ᵉ chunk LLaMA est remplacé par un chunk sur la **licence de Stable Diffusion** ;
- « Quels sont les risques de l'IA générative ? » : des chunks d'autres sections de la page, sur d'autres risques.

Les réponses restent correctes et proches. Pour une question factuelle précise, MMR remplace des chunks pertinents mais redondants par des chunks **moins pertinents** (du bruit) ; pour une question large, il apporte plus de variété. MMR est utile quand le corpus contient beaucoup de passages quasi identiques (doublons, chevauchements), moins sur des questions pointues.

### Question 32 — Mémoire conversationnelle

Sans mémoire, après « Qu'est-ce que LLaMA ? », la question « Et qui l'a créé ? » est cherchée telle quelle : les chunks récupérés viennent de ChatGPT, Stable Diffusion et DALL-E, et la réponse est « Le contexte ne précise pas qui l'a créé. »

**Solution implémentée** (`contextualize_question`) : avant la recherche, le LLM réécrit la question à l'aide des derniers échanges pour qu'elle soit autonome. « Et qui l'a créé ? » devient **« Qui a créé LLaMA ? »** ; les chunks viennent alors de la page LLaMA, et la réponse est « LLaMA a été développé par Meta. ». La réécriture n'est faite que s'il y a un historique (un appel au LLM de plus par question). Elle est activable dans l'interface Streamlit et testée dans le navigateur.

### Question 33 — Citations précises

Avec `build_prompt(citations=True)`, le prompt demande d'indiquer après chaque affirmation le numéro du passage utilisé, et `format_docs` numérote déjà les passages `[1]`, `[2]`… Exemple : « Un GAN utilise deux réseaux entraînés en compétition […] **[1]** Un modèle de diffusion, lui, débruite une image […] **[1]** ». On peut ainsi vérifier chaque affirmation dans la zone repliable des passages.

### Question 34 — Coûts

Mesure avec `tiktoken` (encodage `o200k_base`) sur les 4 questions de la partie 5 : en moyenne **776 tokens en entrée** (consignes + 4 chunks + question) et **28 tokens en sortie**. L'indexation envoie **74 801 tokens** (425 chunks) au modèle d'embeddings.

Tarifs (relevés en ligne, à vérifier sur la page officielle d'OpenAI) : `gpt-6-luna` à 0,10 $ par million de tokens en entrée et 0,50 $ en sortie ; `text-embedding-3-small` à 0,02 $ par million de tokens.
- **Une question** : 776 × 0,10 / 10⁶ + 28 × 0,50 / 10⁶ ≈ **0,00009 $** ; **1 000 questions ≈ 0,09 $** (l'embedding des questions est négligeable).
- **Indexation** : 74 801 × 0,02 / 10⁶ ≈ **0,0015 $** (moins d'un centime, et une seule fois grâce à la persistance).
- **Sans RAG**, envoyer les 12 pages à chaque question (≈ 75 000 tokens) coûterait ≈ 0,0075 $ par question, soit ≈ **7,5 $ pour 1 000 questions** : environ 80 fois plus.

Réserve : si `gpt-6-luna` produit des tokens de raisonnement internes, ils sont facturés comme tokens de sortie et ne sont pas comptés ici.

### Question 35 — Comparaison des fournisseurs

Non réalisée : elle demande une clé Mistral personnelle, que je n'ai pas. Le protocole est prêt : ajouter `MISTRAL_API_KEY` dans `.env`, lancer avec `LLM_PROVIDER=mistral` (une base séparée `chroma_db_mistral` est construite), puis relancer `python experiences.py partie5 q30` et comparer les tableaux (pages récupérées, réponses correctes, temps).

### Question 36 — Modèles locaux

Non réalisée. Démarche : remplacer `get_embeddings()` par `HuggingFaceEmbeddings` (par exemple `intfloat/multilingual-e5-small`, adapté au français) et `get_llm()` par `ChatOllama(model="qwen2.5:3b")`. Attendus : aucun coût par requête et aucune donnée envoyée à l'extérieur, mais des réponses plus lentes sur CPU et de moins bonne qualité en français.

### Question 37 — Évaluation automatique

Méthode proposée, dans l'esprit de RAGAS :
1. **Jeu de test** : une cinquantaine de questions avec leur réponse de référence et le passage source (page et phrase), écrites à la main ou générées par un LLM à partir de passages tirés au hasard, puis relues.
2. **Recherche** : *recall@k* (le passage source, ou la réponse attendue, figure-t-il parmi les k chunks ?) et *MRR* (rang du premier chunk pertinent). C'est ce que mesure la colonne « contexte contient la réponse » d'`experiences.py`.
3. **Génération** :
   - **exactitude** : la réponse contient-elle la réponse de référence (correspondance de mots-clés, comme ici, ou jugement d'un LLM) ;
   - **fidélité** (*faithfulness*) : un LLM juge découpe la réponse en affirmations et vérifie que chacune est justifiée par le contexte ; une affirmation non justifiée est une hallucination ;
   - **pertinence de la réponse** par rapport à la question ;
   - **taux d'abstention justifiée** sur des questions hors corpus, comme (iv).
4. Comparer les configurations (taille des chunks, k, MMR, multi-query) **sur le même jeu de test**, avec plusieurs exécutions, puisque le LLM n'est pas déterministe.

---

## Livrables et étapes restantes

- Code complet : `main.py`, `app.py`, `requirements.txt`, `.gitignore`, `README.md`, `Dockerfile`, `.dockerignore`, `experiences.py`.
- Dépôt GitHub public : <https://github.com/Xelou76/chatbot-rag-ia-generative> (questions 22-23).
- Image Docker construite et testée (question 28).
- **À faire** : déployer sur Streamlit Community Cloud avec `OPENAI_API_KEY` et `APP_PASSWORD` dans les secrets, puis ajouter ici l'URL de l'application (questions 24-26).
