import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_objectbox.vectorstores import ObjectBox
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title('RAG with LLAMA3, GROQ INFERENCING ENGINE and Objectbox')


llm = ChatGroq(api_key=groq_api_key,model='llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
"""
You are an helpful information retrival assistant.
Context provided to you are through uploaded pdfs and retriever used form objectbox vectorstore.
Answer the question based on provided context only.
If question is not available in context then say "I can not help you with this case".
Please provide most accurate response based on context only.
<context>
{context}
</context>
Question: {input}
"""
)

output_dir = "docs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(output_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"File Uploaded successfully")
    st.write("Uploaded files", os.listdir(output_dir))
       

if st.button("Check Present files"):
    if os.listdir(output_dir):
        st.write("Contents of folder are:")
        for file_name in os.listdir(output_dir):
            st.write(file_name)
    else:
        st.write("No uploaded files")

if st.button("Clear Uploaded content"):
    if os.listdir(output_dir):
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)
        st.success("Cleared Uploaded content successfully")
    else:
        st.write("Upload folder is already empty.")


def vector_embedding():
    
    if 'vector' not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader('./docs')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.final_docs = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100).split_documents(st.session_state.docs)
        st.session_state.vector = ObjectBox.from_documents(st.session_state.final_docs,st.session_state.embeddings,embedding_dimensions=512)


if st.button('Use data from current pdfs'):
    if os.listdir(output_dir):  
        try:
            vector_embedding()
            st.success('Document data read succesfully')
        except Exception as e:
            st.error(e)
    else:
        st.write("Please upload PDF first")

inp_prom = st.text_input('Enter prompt to search from Documents')


if inp_prom:
    try:
        start = time.process_time()
        document_chain = create_stuff_documents_chain(llm,prompt)
        retriever = st.session_state.vector.as_retriever()
        retriever_chain = create_retrieval_chain(retriever,document_chain)
        response = retriever_chain.invoke({'input':inp_prom})
        st.write(response['answer'])
        with st.expander('Check source of answer'):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('---------------------------------------------')
        st.success(f'Response in {time.process_time()-start}')
    except Exception as e:
        st.error(e)