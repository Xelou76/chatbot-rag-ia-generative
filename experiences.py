"""
Expériences du compte rendu (parties 5 et bonus). Les résultats sont écrits dans resultats/.
Usage : python experiences.py partie5 | q17 | q18 | q29 | q30 | q31 | q32 | q33 | q34 | tout
"""

import sys
import time
import unicodedata

import tiktoken

from main import (CHAT_MODEL, EMBEDDING_MODEL, build_prompt, format_docs, generate_queries, get_retriever,
                  get_vectorstore, rag_answer)

PARTIE5 = [
    "Qui a développé Stable Diffusion ?",
    "Quelle est la différence entre un GAN et un modèle de diffusion ?",
    "En quelle année ChatGPT a-t-il été lancé ?",
    "Quelle est la capitale de l'Australie ?",
]

# Questions de référence (bonus 29, 30, 31) : (question, mots-clés dont l'un doit figurer dans la réponse)
REFERENCE = [
    ("Qui a développé Stable Diffusion ?", ["stability"]),
    ("En quelle année ChatGPT a-t-il été lancé ?", ["2022"]),
    ("Qui a inventé les réseaux antagonistes génératifs ?", ["goodfellow"]),
    ("Qui a fondé Midjourney ?", ["holz"]),
    ("Quelle entreprise développe Gemini ?", ["google"]),
]


def normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def contains(text, keywords):
    return any(normalize(k) in normalize(text) for k in keywords)


def show_docs(docs, out):
    for i, d in enumerate(docs, start=1):
        out(f"    [{i}] {d.metadata['title']} (car. {d.metadata.get('start_index')}) : "
            f"{d.page_content[:220].replace(chr(10), ' ')}...")


class Report:
    def __init__(self, name):
        self.file = open(f"resultats/{name}.txt", "w", encoding="utf-8")

    def __call__(self, text=""):
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()


def partie5():
    out = Report("partie5_q16")
    retriever = get_retriever(verbose=False)
    for q in PARTIE5:
        t0 = time.time()
        answer, docs = rag_answer(q, retriever, stream=False)
        out(f"### {q}   ({time.time() - t0:.1f} s)")
        out(answer)
        out("Sources : " + ", ".join(dict.fromkeys(d.metadata["source"] for d in docs)))
        show_docs(docs, out)
        out()


def q17():
    out = Report("q17_sans_consigne")
    retriever = get_retriever(verbose=False)
    q = "Quelle est la capitale de l'Australie ?"
    for name, prompt in [("prompt complet", build_prompt()),
                         ("sans « uniquement à partir du contexte »", build_prompt(only_context=False)),
                         ("sans aucune des deux consignes", build_prompt(only_context=False, say_unknown=False))]:
        answer, _ = rag_answer(q, retriever, stream=False, prompt=prompt)
        out(f"### {name}\n{answer}\n")


def q18():
    out = Report("q18_mauvaises_reponses")
    retriever = get_retriever(verbose=False)
    candidates = [
        "Quels sont les modèles de génération d'images présentés dans le corpus et qui les a développés ?",
        "Combien de paramètres compte le plus grand modèle LLaMA 2 ?",
        "Quelle est la date de sortie de DALL-E 3 ?",
        "Qui sont les fondateurs d'OpenAI ?",
        "Quel algorithme d'apprentissage par renforcement est utilisé pour entraîner ChatGPT ?",
        "Lequel est sorti en premier : Midjourney ou Stable Diffusion ?",
    ]
    for q in candidates:
        answer, docs = rag_answer(q, retriever, stream=False)
        out(f"### {q}\n{answer}")
        show_docs(docs, out)
        out()


def q29():
    out = Report("q29_multi_query")
    retriever = get_retriever(verbose=False)
    for q, keywords in REFERENCE + [("Qui a créé LLaMA ?", ["meta"]),
                                    ("Comment fonctionne l'entraînement adversarial ?", ["generateur", "discriminateur"])]:
        out(f"### {q}")
        out("  reformulations : " + " | ".join(generate_queries(q)))
        for name, multi in [("simple", False), ("multi-query + RRF", True)]:
            t0 = time.time()
            answer, docs = rag_answer(q, retriever, stream=False, multi_query=multi)
            out(f"  [{name}] {time.time() - t0:.1f} s · contexte contient la réponse : "
                f"{contains(format_docs(docs), keywords)} · réponse correcte : {contains(answer, keywords)}")
            out("    pages : " + ", ".join(d.metadata["title"] for d in docs))
            out("    réponse : " + answer.replace("\n", " ")[:300])
        out()


def q30():
    out = Report("q30_parametres")
    out("chunk_size | k | contexte contient la réponse | réponse correcte | tokens de contexte (moyenne) | temps (s)")
    enc = tiktoken.get_encoding("o200k_base")
    for chunk_size in (300, 1000, 2000):
        overlap = chunk_size // 5
        t0 = time.time()
        get_vectorstore(chunk_size, overlap, verbose=True)
        out(f"(index chunk_size={chunk_size}, overlap={overlap} prêt en {time.time() - t0:.0f} s)")
        for k in (2, 4, 8):
            retriever = get_retriever(k=k, chunk_size=chunk_size, chunk_overlap=overlap, verbose=False)
            hits = correct = tokens = 0
            t0 = time.time()
            for q, keywords in REFERENCE:
                answer, docs = rag_answer(q, retriever, stream=False)
                hits += contains(format_docs(docs), keywords)
                correct += contains(answer, keywords)
                tokens += len(enc.encode(format_docs(docs)))
            out(f"{chunk_size} | {k} | {hits}/5 | {correct}/5 | {tokens / 5:.0f} | {(time.time() - t0) / 5:.1f}")


def q31():
    out = Report("q31_mmr")
    sim = get_retriever(verbose=False)
    mmr = get_retriever(search_type="mmr", verbose=False)
    for q in ["Qui a créé LLaMA ?", "Qu'est-ce qu'un GAN ?", "Quels sont les risques de l'IA générative ?",
              "Qui a développé Stable Diffusion ?"]:
        out(f"### {q}")
        for name, r in [("similarité", sim), ("MMR", mmr)]:
            docs = r.invoke(q)
            out(f"  {name:<10} pages : " + ", ".join(f"{d.metadata['title']} ({d.metadata['start_index']})" for d in docs))
        answer_sim, _ = rag_answer(q, sim, stream=False)
        answer_mmr, _ = rag_answer(q, mmr, stream=False)
        out("  réponse similarité : " + answer_sim.replace("\n", " ")[:300])
        out("  réponse MMR        : " + answer_mmr.replace("\n", " ")[:300])
        out()


def q32():
    out = Report("q32_memoire")
    retriever = get_retriever(verbose=False)
    first = "Qu'est-ce que LLaMA ?"
    answer1, _ = rag_answer(first, retriever, stream=False)
    out(f"### {first}\n{answer1}\n")
    follow = "Et qui l'a créé ?"
    answer2, docs2 = rag_answer(follow, retriever, stream=False)
    out(f"### {follow} (sans mémoire)\n{answer2}\n    pages : " + ", ".join(d.metadata["title"] for d in docs2) + "\n")
    history = [("user", first), ("assistant", answer1)]
    from main import contextualize_question
    out(f"Question reformulée : {contextualize_question(follow, history)}")
    answer3, docs3 = rag_answer(follow, retriever, stream=False, history=history)
    out(f"### {follow} (avec mémoire)\n{answer3}\n    pages : " + ", ".join(d.metadata["title"] for d in docs3))


def q33():
    out = Report("q33_citations")
    retriever = get_retriever(verbose=False)
    for q in ["Quelle est la différence entre un GAN et un modèle de diffusion ?", "Qui a développé Stable Diffusion ?"]:
        answer, docs = rag_answer(q, retriever, stream=False, prompt=build_prompt(citations=True))
        out(f"### {q}\n{answer}")
        show_docs(docs, out)
        out()


def q34():
    out = Report("q34_couts")
    enc = tiktoken.get_encoding("o200k_base")
    retriever = get_retriever(verbose=False)
    totals = []
    for q in PARTIE5:
        docs = retriever.invoke(q)
        messages = build_prompt().format_messages(context=format_docs(docs), question=q)
        n_in = sum(len(enc.encode(m.content)) for m in messages) + 4 * len(messages)  # + quelques tokens de structure
        answer, _ = rag_answer(q, retriever, stream=False)
        n_out = len(enc.encode(answer))
        totals.append((n_in, n_out))
        out(f"{q} : {n_in} tokens en entrée, {n_out} tokens en sortie")
    mean_in = sum(t[0] for t in totals) / len(totals)
    mean_out = sum(t[1] for t in totals) / len(totals)
    out(f"Moyenne : {mean_in:.0f} tokens en entrée, {mean_out:.0f} en sortie, modèle {CHAT_MODEL}")
    chunks = get_vectorstore(verbose=False).get()["documents"]
    enc_emb = tiktoken.get_encoding("cl100k_base")
    n_index = sum(len(enc_emb.encode(c)) for c in chunks)
    out(f"Indexation : {len(chunks)} chunks, {n_index} tokens envoyés à {EMBEDDING_MODEL}")


def q10():
    """BM25 contre embeddings : rang du premier chunk qui contient l'information attendue."""
    import re
    from rank_bm25 import BM25Okapi
    out = Report("q10_bm25_vs_embeddings")
    vectorstore = get_vectorstore(verbose=False)
    data = vectorstore.get()
    texts = data["documents"]

    def tokenize(text):
        return re.findall(r"\w+", normalize(text))

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    cases = [
        # paraphrase : aucun mot commun avec le passage (« dessine », « phrase » ≠ « génère des images », « descriptions textuelles »)
        ("Quel logiciel d'OpenAI dessine à partir d'une phrase ?", "DALL-E (ou DALL·E), prononcé"),
        # question en anglais sur un corpus français
        ("Who invented generative adversarial networks?", "Ian Goodfellow"),
        # identifiants rares et exacts : noms propres cités une seule fois dans le corpus
        ("Que fait Emad Mostaque ?", "Emad Mostaque"),
        ("Que dit William Orrick ?", "Orrick"),
        ("Qui est Greg Brockman ?", "Brockman"),
        # identifiants rares mais très présents dans leur passage : les deux méthodes réussissent
        ("Que permet la licence CreativeML OpenRAIL-M ?", "CreativeML OpenRAIL-M"),
        ("Qu'est-ce que SDXL-Turbo ?", "SDXL-Turbo"),
    ]
    out("question | rang BM25 | rang embeddings (1 = premier, >20 = absent du top 20)")
    for question, marker in cases:
        gold = {i for i, t in enumerate(texts) if marker in t}
        scores = bm25.get_scores(tokenize(question))
        bm25_rank = sorted(range(len(texts)), key=lambda i: -scores[i])
        dense = vectorstore.similarity_search(question, k=20)
        dense_rank = [texts.index(d.page_content) for d in dense]

        def first(ranking):
            return next((r for r, i in enumerate(ranking[:20], start=1) if i in gold), ">20")

        out(f"{question} | {first(bm25_rank)} | {first(dense_rank)}")


EXPERIENCES = {"q10": q10, "partie5": partie5, "q17": q17, "q18": q18, "q29": q29, "q30": q30, "q31": q31,
               "q32": q32, "q33": q33, "q34": q34}

if __name__ == "__main__":
    names = sys.argv[1:] or ["tout"]
    for name in (EXPERIENCES if names == ["tout"] else names):
        print(f"\n========== {name} ==========")
        EXPERIENCES[name]()
