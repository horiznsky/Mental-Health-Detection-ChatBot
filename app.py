import streamlit as st
import os
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

st.set_page_config(page_title="Serenity AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>

/* App background */
.main {
    background-color: #0f172a; /* deep navy */
}

/* Chat container */
.chat-container {
    border-radius: 12px;
    padding: 20px;
    background-color: #1e293b; /* slate blue */
    box-shadow: 0 6px 12px rgba(0,0,0,0.4);
}

/* Bot message */
.bot-msg {
    background-color: #1e3a5f; /* calm blue */
    padding: 15px;
    border-radius: 15px;
    border-top-left-radius: 0;
    margin-bottom: 10px;
    color: #e0f2fe; /* soft cyan text */
    border: 1px solid #334155;
}

/* User message */
.user-msg {
    background-color: #1f3d2b; /* muted green */
    padding: 15px;
    border-radius: 15px;
    border-top-right-radius: 0;
    margin-bottom: 10px;
    text-align: right;
    color: #dcfce7; /* soft mint text */
    border: 1px solid #14532d;
}

/* Title */
h1 {
    color: #e5e7eb; /* off-white */
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #020617; /* darker navy */
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #1e293b;
    color: #c7d2fe;
    border-radius: 8px;
}

/* Input box */
textarea, input {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border: 1px solid #334155 !important;
}

/* Spinner text */
.css-1kyxreq {
    color: #93c5fd;
}

</style>
""", unsafe_allow_html=True)


st.title(" Serenity: Professional Mental Health Support")

@st.cache_resource
def load_resources():
    try:
        print("Loading Vector DB...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
        
        print("Loading Llama Model...")
        if not os.path.exists("./models/model_final.gguf"):
            raise FileNotFoundError("Model file not found! Check ./models/model_final.gguf")
            
        llm = LlamaCpp(
            model_path="./models/model_final.gguf",
            n_gpu_layers=20, 
            n_ctx=4096,
            temperature=0.3,
            verbose=True
        )
        return db, llm
    except Exception as e:
        return None, None

db, llm = load_resources()

if db is None or llm is None:
    st.error(" System failed to load. Check terminal for details.")
    st.stop()
else:
    st.sidebar.success(" AI System Online")

retriever = db.as_retriever(search_kwargs={"k": 3})

template = """
System: You are a professional mental health assistant trained on WHO guidelines.
Use the following pieces of medical context to answer the user's question safely.
If the answer is not in the context, say "I don't have clinical information on that specific topic, but I am here to listen."
Do not provide prescriptions or medical diagnosis.

Context: {context}

User: {question}

Assistant:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)
        if "sources" in msg:
            with st.expander(" View Clinical Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='user-msg'>{user_input}</div>", unsafe_allow_html=True)

    with st.spinner("Generating Response..."):
        try:
            result = qa_chain.invoke({"query": user_input})
            response_text = result["result"]
            source_docs = result["source_documents"]
            sources_text = [f"Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})" for doc in source_docs]

            st.markdown(f"<div class='bot-msg'>{response_text}</div>", unsafe_allow_html=True)
            with st.expander(" View Sources"):
                for s in sources_text:
                    st.write(s)

            st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": sources_text})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")

st.sidebar.title("System Specs")
st.sidebar.info("Model: Llama-3.2-3B (Quantized)")
st.sidebar.info("Knowledge Base: WHO mhGAP Guidelines + NIMH FAQ")
st.sidebar.info(f"Hardware: RTX 3050 (GPU Accelerated)")