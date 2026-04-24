import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

import wikipediaapi
import warnings
warnings.filterwarnings("ignore")

# Page config

st.set_page_config(
    page_title="AI/ML Knowledge Chatbot",
    layout="wide"
)

# Wikipedia topics 

WIKI_TOPICS = [
    "Machine learning",
    "Deep learning",
    "Natural language processing",
    "Transformer (deep learning architecture)",
    "Convolutional neural network",
    "Recurrent neural network",
    "Reinforcement learning",
    "Generative adversarial network",
    "BERT (language model)",
    "Large language model",
    "Retrieval-augmented generation",
]

# Knowledge base builder

@st.cache_resource(show_spinner=False)
def build_vectorstore():
    status = st.empty()

    # Fetch Wikipedia pages
    status.info("Fetching Wikipedia pages...")
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="RAG-Chatbot/1.0"
    )

    docs = []
    for topic in WIKI_TOPICS:
        page = wiki.page(topic)
        if page.exists():
            # Store the source title as metadata
            docs.append(Document(
                page_content=page.text,
                metadata={"source": topic, "url": page.fullurl}
            ))

    status.info(f"Loaded {len(docs)} Wikipedia articles. Chunking...")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    status.info(f"Split into {len(chunks)} chunks. Building embeddings...")

    # Embed and index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    status.success(f"Vector store ready — {len(chunks)} chunks indexed.")
    status.empty()

    return vectorstore


@st.cache_resource(show_spinner=False)
def build_chain(_vectorstore):
 
    groq_api_key = os.environ.get("GROQ_API_KEY", "")

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,   
        max_tokens=1024
    )

    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4} 
    )

    # Rephrase the user question using chat history 
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and a new user question, "
         "rewrite the question to be fully self-contained. "
         "If the question is already self-contained, return it unchanged. "
         "Do NOT answer the question — only rewrite it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answer the question using retrieved context 
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI/ML knowledge assistant. "
         "Use the retrieved context below to answer the user's question accurately. "
         "If the answer is not in the context, say you don't know — do not make things up. "
         "Keep answers concise and clear.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combines retrieval + QA into one runnable
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Wrap with message history 
    # RunnableWithMessageHistory injects the stored chat history on every call
    # and appends the new exchange to the store automatically
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,           
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return chain_with_history


# Chat history store 
store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Session state 

if "messages" not in st.session_state:
    st.session_state.messages = []        

if "chain" not in st.session_state:
    st.session_state.chain = None

if "session_id" not in st.session_state:
    st.session_state.session_id = "default"  

# Sidebar 

with st.sidebar:
    st.title(" Knowledge Base")
    st.markdown("This chatbot knows about:")
    for topic in WIKI_TOPICS:
        st.markdown(f"- {topic}")

    st.markdown("---")

    if st.button(" Clear Chat History"):
        st.session_state.messages = []
        # Clear the in-memory message store for this session
        if st.session_state.session_id in store:
            del store[st.session_state.session_id]
        st.rerun()

    st.markdown("---")
    st.markdown(
        "**Stack:** Groq LLaMA3 · FAISS · "
        "sentence-transformers · LangChain · Streamlit"
    )



st.title("AI/ML Knowledge Chatbot")
st.markdown(
    "Ask me anything about Machine Learning, Deep Learning, NLP, Transformers, "
    "and more."
)
st.markdown("---")


groq_api_key = os.environ.get("GROQ_API_KEY", "")
if not groq_api_key:
    st.error(
        " GROQ_API_KEY environment variable not set. "
    )
    st.stop()

# Build the chain once on first load
if st.session_state.chain is None:
    with st.spinner("Building knowledge base — this takes ~30 seconds on first load..."):
        try:
            vectorstore = build_vectorstore()
            st.session_state.chain = build_chain(vectorstore)
            st.success(" Ready! Ask me anything.")
        except Exception as e:
            st.error(f"Failed to initialise: {e}")
            st.stop()

# Chat history display 

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(" Sources retrieved"):
                for src in message["sources"]:
                    st.markdown(f"- **{src}**")
 
# Chat input

if prompt := st.chat_input("Ask about any AI/ML concept..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
                answer = result["answer"]

                # Deduplicate source titles for the citation expander
                sources = list({
                    doc.metadata["source"]
                    for doc in result.get("context", [])
                })

                st.markdown(answer)

                if sources:
                    with st.expander(" Sources retrieved"):
                        for src in sources:
                            st.markdown(f"- **{src}**")

                st.session_state.messages.append({
                    "role"    : "assistant",
                    "content" : answer,
                    "sources" : sources
                })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg, "sources": []
                })