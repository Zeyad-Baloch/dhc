## Task: Context-Aware RAG Chatbot (`rag_app.py`)

**Objective:** Build a conversational chatbot that retrieves answers from a vectorized knowledge base and maintains conversation history across turns.

**Knowledge base:** Wikipedia articles on AI/ML concepts fetched programmatically via `wikipedia-api`.

**Stack:** Groq LLaMA3-8b · FAISS · sentence-transformers · LangChain · Streamlit

**Working:**
- Fetched Wikipedia pages at runtime and wrapped them as `Document` objects with source metadata.
- Split into 500-token chunks with 50-token overlap using `RecursiveCharacterTextSplitter`.
- Embedded all chunks with `sentence-transformers/all-MiniLM-L6-v2` and indexed into FAISS.
- Built a `history_aware_retriever` that rewrites the user's question using chat history before querying FAISS, so follow-up questions like "explain more" retrieve the right documents rather than searching literally.
- Passed retrieved chunks + conversation history into a QA prompt.
- Wrapped the full chain in `RunnableWithMessageHistory` so history is injected on every call.
- Deployed as a Streamlit app.


**Notes:**
- Set `GROQ_API_KEY` as an environment variable before running.`
- The vector store builds on first launch (~30 seconds) and is cached for the rest of the session
- Run with: `streamlit run rag_app.py`

---

```bash
pip install streamlit langchain-classic langchain-groq langchain-core
pip install langchain-text-splitters langchain-community faiss-cpu
pip install sentence-transformers wikipedia-api
export GROQ_API_KEY=your_key_here
```

