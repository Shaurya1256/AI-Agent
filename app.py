import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="AI PDF Q&A Assistant", layout="wide", page_icon="📚")

# Custom CSS for hover effect and better UI
st.markdown("""
<style>
    .stButton>button:hover {
        background-color: #4B9CD3;
        color: white;
        border-radius: 10px;
        transition: 0.3s;
    }
    .chat-box {
        background-color: #f1f1f1;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-msg {
        background-color: #DCF8C6;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 5px;
    }
    .bot-msg {
        background-color: #E6E6FA;
        padding: 8px;
        border-radius: 8px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: darkblue;'>📚 AI PDF Q&A Assistant</h1>", unsafe_allow_html=True)
st.write("Upload a PDF and ask questions. The AI will answer based on the document.")

# Initialize session state variables
if 'db' not in st.session_state:
    st.session_state['db'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Process PDF
if uploaded_file:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF... This may take a while for large files."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state['db'] = FAISS.from_documents(chunks, embeddings)
            
            st.success("✅ PDF processed! Now you can ask questions.")

# Question input and submit
if st.session_state['db'] is not None:
    user_question = st.text_input("Ask a question about the PDF:")

    if st.button("Submit Question"):
        if user_question.strip() == "":
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Generating answer..."):
                retriever = st.session_state['db'].as_retriever()
                docs = retriever.get_relevant_documents(user_question)
                context = " ".join([d.page_content for d in docs[:2]])
                prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {user_question}\nAnswer:"
                
                qa_pipeline = pipeline("text-generation", model="distilgpt2")
                result = qa_pipeline(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']

                # Save to chat history
                st.session_state['chat_history'].append({"question": user_question, "answer": result})

# Display chat history
if st.session_state['chat_history']:
    st.markdown("### 💬 Chat History")
    for i, chat in enumerate(st.session_state['chat_history'][::-1]):  # show latest first
        st.markdown(f"<div class='user-msg'><b>Q:</b> {chat['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bot-msg'><b>A:</b> {chat['answer']}</div>", unsafe_allow_html=True)
        if st.button(f"Ask Again ❓", key=f"repeat_{i}"):
            st.text_input("Ask a question about the PDF:", value=chat['question'])
 


        
