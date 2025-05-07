# Increff-AI-Support-Chatbot



An end‑to‑end, Retrieval‑Augmented Generation (RAG) customer‑support chatbot for an electronics e‑commerce store.
It stores FAQs in `data/faq.json`, converts the questions into OpenAI embeddings, and indexes them with FAISS for millisecond similarity search.
At runtime the bot retrieves the top‑k relevant FAQ snippets, stitches them—along with the last few user‑bot turns—into a concise prompt, and generates grounded answers with GPT‑3.5‑Turbo.
The solution includes a Streamlit web UI, a CLI mode, automated pytest coverage for retrieval and multi‑turn coherence, and clear guardrails for ambiguity, out‑of‑scope queries, and hallucination control.



## 🗺️ End‑to‑End Process & Steps

| #  | Phase                               | What We Did                                                                                                                                                                                                                           | Key Files / Commands                                                               |
| -- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 1  | **Project Scaffold**                | Created a clean repo layout:<br>`ai_support_chatbot/` → `data/`, `utils/`, `tests/`, `streamlit_app.py`, etc.                                                                                                                         | —                                                                                  |
| 2  | **Dependency Pinning**              | Added `requirements.txt` with **OpenAI**, **faiss‑cpu**, **streamlit**, **pytest**, **python‑dotenv**.                                                                                                                                | `pip install -r requirements.txt`                                                  |
| 3  | **FAQ Authoring**                   | Expanded **`data/faq.json`** to \~15 high‑quality Q‑A pairs covering returns, warranty, shipping, security, payment.                                                                                                                  | —                                                                                  |
| 4  | **Embedding & Vector Store**        | *Script*: **`embed_faq.py`**<br>• Reads `faq.json`<br>• Batches questions into **OpenAI `text‑embedding‑3-small`**<br>• Normalizes vectors → **FAISS inner‑product** index<br>• Persists `faq_faiss.index` + metadata pickle.         | `bash\npython embed_faq.py\n`                                                      |
| 5  | **Conversation Memory**             | Implemented **deque‑based** history buffer (`utils/memory.py`, default 6 turns) to preserve context across multi‑turn chats.                                                                                                          | —                                                                                  |
| 6  | **Retrieval + Generation Pipeline** | *Module*: **`chatbot.py`**<br>• Embeds user query with same OpenAI model.<br>• Retrieves top‑3 FAQ snippets from FAISS.<br>• Builds lean prompt = `FAQ context + history + user msg`.<br>• Calls **GPT‑3.5‑Turbo** `temperature=0.3`. | `python\nfrom chatbot import chat_once\nchat_once(\"How do I return a phone?\")\n` |
| 7  | **Streamlit UI**                    | *App*: **`streamlit_app.py`**<br>• Minimal chat front‑end.<br>• Stores chat log in `st.session_state`.<br>• Reset button clears history.                                                                                              | `bash\nstreamlit run streamlit_app.py\n`                                           |
| 8  | **CLI Utility**                     | Running `python chatbot.py` opens a REPL for quick local testing without UI overhead.                                                                                                                                                 | —                                                                                  |
| 9  | **Automated Tests**                 | *Tests*: **`tests/test_chatbot.py`**<br>• Retrieval correctness (keywords found).<br>• Multi‑turn follow‑up resolution test.                                                                                                          | `pytest -q`                                                                        |
| 10 | **Edge‑Case Strategy**              | Documented 10 edge cases—ambiguous queries, OOS topics, token overflow, etc.—and baked mitigations into prompt + code.                                                                                                                | See README Edge‑Case table                                                         |
| 11 | **Prompt Engineering**              | - Top‑3 FAQ snippets only.<br>- History capped.<br>- Explicit guardrails: “Answer ONLY from FAQ context.”<br>- Low temperature for determinism.                                                                                       | —                                                                                  |
| 12 | **PDF & Docs (Optional)**           | Generated a solution PDF summarizing architecture & steps (see `AI_Intern_Assignment_Solution_Gopesh.pdf`).                                                                                                                           | —                                                                                  |

---

## 🏗️ Architecture & Flow Diagram

```mermaid
flowchart TD
    A[User<br>(Web / CLI)] -->|sends question| B(Streamlit Front‑end)
    B -->|API call| C[chat_once()]
    C --> D[embed_query → OpenAI Embeddings]
    D --> E[FAISS Search<br>(top‑3)]
    E --> F[Build Prompt<br>FAQ ctx + history + user msg]
    F --> G[GPT‑3.5‑Turbo]
    G -->|response| H[ConversationMemory<br>append]
    H -->|final answer| B
```

---

## 🔧 Setup & Run

```bash
# 1. Clone & enter
git clone <repo-url>
cd ai_support_chatbot

# 2. Dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure OpenAI key
cp .env.example .env          # edit with your key

# 4. Build vector index
python embed_faq.py

# 5. Launch Web UI
streamlit run streamlit_app.py
```

---

## 🧪 Testing

```bash
pytest -q
# All tests should pass:
# 3 passed in <1.0s
```

*Tests cover retrieval accuracy and multi‑turn context retention.*
