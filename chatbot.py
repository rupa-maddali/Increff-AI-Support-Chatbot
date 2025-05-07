import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from utils.memory import ConversationMemory

load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
INDEX_PATH = "data/faq_faiss.index"
META_PATH = "data/faq_meta.pkl"
LLM_MODEL = "gpt-3.5-turbo"
TOP_K = 3
MIN_SCORE = 0.7  # Minimum similarity score to consider a match
memory = ConversationMemory(max_turns=6)

# Initialize OpenAI client

os.environ["USER_AGENT"] = "MyApp/1.0 (+https://example.com)"
os.environ[
    "OPENAI_API_KEY"] = "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"  # Replace with your actual key




client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Load persistent data once ──────────────────────────────────────────────
try:
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    questions, answers = meta["questions"], meta["answers"]
except Exception as e:
    raise RuntimeError(f"Failed to load FAQ data: {str(e)}")


# ─── Embedding helper ───────────────────────────────────────────────────────
def embed_query(text: str) -> np.ndarray:
    """Generate normalized embedding for a query text"""
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[text]
        )
        em = response.data[0].embedding
        v = np.asarray(em, dtype="float32")
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {str(e)}")


# ─── Retrieval + generation ────────────────────────────────────────────────
def retrieve(user_msg: str) -> list[tuple[str, str]]:
    """Retrieve relevant FAQ entries with similarity filtering"""
    try:
        v = embed_query(user_msg).reshape(1, -1)
        scores, ids = index.search(v, TOP_K)

        # Filter results by similarity score and return with scores
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if score >= MIN_SCORE and 0 <= idx < len(questions):
                results.append((questions[idx], answers[idx], score))

        # Sort by score (descending) and return only Q&A pairs
        results.sort(key=lambda x: x[2], reverse=True)
        return [(q, a) for q, a, _ in results]
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return []


# def build_prompt(user_msg: str, context: list[tuple[str, str]]) -> str:
#     """Construct the LLM prompt with context and conversation history"""
#     # Handle case where no relevant context was found
#     if not context:
#         context = [("No relevant FAQ found", "I don't have information about this.")]
#
#     faq_block = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in context])
#     history = memory.format()
#
#     return f"""You are an AI assistant for an electronics e-commerce company.
# When answering, prioritize these rules:
# 1. Use ONLY the FAQ context below when available
# 2. If the question is unclear, ask for clarification
# 3. If you don't know, say so honestly
# 4. Keep responses concise and professional
#
# FAQ context:
# {faq_block}
#
# Conversation history:
# {history}
#
# User: {user_msg}
# Bot:"""


def build_prompt(user_msg: str, context: list[tuple[str, str]]) -> str:
    """Construct the LLM prompt with context and conversation history"""
    # Detect if this is small talk/greeting
    is_small_talk = any(word in user_msg.lower() for word in
                        ["hi", "hello", "hey", "how are you", "what's up"])

    # Handle small talk differently
    if is_small_talk:
        return f"""You are a friendly AI assistant for an electronics e-commerce company.
The user has engaged in small talk. Respond naturally but keep it brief and professional.

Conversation history:
{memory.format()}

User: {user_msg}
Bot:"""

    # Normal FAQ-based response
    if not context:
        context = [("No relevant FAQ found",
                    "I don't have information about this in our knowledge base.")]

    faq_block = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in context])
    history = memory.format()

    return f"""You are an AI assistant for an electronics e-commerce company. 
When answering, follow these rules:
1. For technical/product questions, use ONLY the FAQ context below
2. For general questions, answer briefly based on your general knowledge
3. If unsure, say you don't know
4. Keep responses professional but friendly

FAQ context:
{faq_block}

Conversation history:
{history}

User: {user_msg}
Bot:"""




def chat_once(user_msg: str) -> str:
    """Process one user message through the full pipeline"""
    if not user_msg.strip():
        return "Please provide a valid question or request."

    try:
        # Retrieve relevant context
        context = retrieve(user_msg)

        # Build and send prompt
        prompt = build_prompt(user_msg, context)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=256
        )

        # Process and store response
        bot_msg = response.choices[0].message.content.strip()
        memory.add(user_msg, bot_msg)
        return bot_msg
    except Exception as e:
        print(f"Error in chat processing: {str(e)}")
        return "Sorry, I encountered an error processing your request."


# CLI quick test -------------------------------------------------------------
if __name__ == "__main__":
    print("AI Support Chatbot (type 'exit' to quit)\n")
    while True:
        try:
            u = input("You: ").strip()
            if u.lower() in {"exit", "quit", "bye"}:
                print("Bot: Goodbye! Have a great day.")
                break
            if not u:
                continue
            print("Bot:", chat_once(u))
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"System error: {str(e)}")
            break