# import streamlit as st
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_groq import ChatGroq
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Streamlit app
# st.title("Chat with Your PDF 📄")
# st.write("Upload a PDF file and chat with its content using Groq!")

# # Input the Groq API Key
# api_key = st.text_input("Enter your Groq API key:", type="password", value=os.getenv("GROQ_API_KEY"))

# # Check if Groq API key is provided
# if api_key:
#     llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

#     # Manage chat history
#     if "store" not in st.session_state:
#         st.session_state.store = {}

#     session_id = st.text_input("Session ID", value="default_session")

#     # File uploader for PDF
#     uploaded_files = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)
#     if uploaded_files:
#         documents = []
#         for uploaded_file in uploaded_files:
#             with open(f"./temp_{uploaded_file.name}", "wb") as temp_pdf:
#                 temp_pdf.write(uploaded_file.read())
            
#             loader = PyPDFLoader(f"./temp_{uploaded_file.name}")
#             docs = loader.load()
#             documents.extend(docs)

#         # Split documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.split_documents(documents)

#         # Create vectorstore
#         vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

#         # Create retriever
#         retriever = vectorstore.as_retriever()

#         # History-aware retriever
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question that can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#         # QA system
#         qa_system_prompt = (
#             "You are an assistant for answering questions based on documents. "
#             "Use the provided context to answer the question concisely. If you "
#             "don't know the answer, say you don't know.\n\n{context}"
#         )
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         # Session history
#         def get_session_history(session: str) -> ChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id] = ChatMessageHistory()
#             return st.session_state.store[session_id]

#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )

#         # User interaction
#         user_input = st.text_input("Ask a question about your PDF:")
#         if user_input:
#             session_history = get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={"configurable": {"session_id": session_id}},
#             )
#             st.write("Assistant:", response["answer"])
#             st.write("Chat History:", session_history.messages)
# else:
#     st.warning("Please provide your Groq API Key.")



# import os
# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_groq import ChatGroq
# from langchain_core.runnables.history import RunnableWithMessageHistory

# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Streamlit app
# st.title("Chat with Your PDF 📄")
# st.write("Upload a PDF file and chat with its content using Groq!")

# # Input the Groq API Key
# api_key = st.text_input("Enter your Groq API key:", type="password", value=os.getenv("GROQ_API_KEY"))

# # Check if Groq API key is provided
# if api_key:
#     #llm = ChatGroq(groq_api_key=api_key, model_name="Mixtral-8x7b-32768")
#     llm = ChatGroq(groq_api_key=api_key, model_name="Deepseek-R1-Distill-Llama-70b")

#     # Manage chat history
#     if "store" not in st.session_state:
#         st.session_state.store = {}

#     session_id = st.text_input("Session ID", value="default_session")

#     # File uploader for PDF
#     uploaded_files = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)
#     if uploaded_files:
#         documents = []
#         for uploaded_file in uploaded_files:
#             with open(f"./temp_{uploaded_file.name}", "wb") as temp_pdf:
#                 temp_pdf.write(uploaded_file.read())
            
#             loader = PyPDFLoader(f"./temp_{uploaded_file.name}")
#             docs = loader.load()
#             documents.extend(docs)

#         # Split documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.split_documents(documents)

#         # Create FAISS vectorstore
#         vectorstore = FAISS.from_documents(chunks, embeddings)

#         # Create retriever
#         retriever = vectorstore.as_retriever()

#         # History-aware retriever
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question that can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#         # QA system
#         qa_system_prompt = (
#             "You are an assistant for answering questions based on documents. "
#             "Use the provided context to answer the question concisely. If you "
#             "don't know the answer, say you don't know.\n\n{context}"
#         )
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         # Session history
#         def get_session_history(session: str) -> ChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id] = ChatMessageHistory()
#             return st.session_state.store[session_id]

#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )

#         # User interaction
#         user_input = st.text_input("Ask a question about your PDF:")
#         if user_input:
#             session_history = get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={"configurable": {"session_id": session_id}},
#             )
#             st.write("Assistant:", response["answer"])
#             st.write("Chat History:", session_history.messages)
# else:
#     st.warning("Please provide your Groq API Key.")


# import os
# import streamlit as st
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_groq import ChatGroq
# from langchain_core.runnables.history import RunnableWithMessageHistory

# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Streamlit app
# st.title("Chat with Your PDF 📄")
# st.write("Upload a PDF file and chat with its content using Groq!")

# # Input the Groq API Key
# api_key = st.text_input("Enter your Groq API key:", type="password", value=os.getenv("GROQ_API_KEY"))

# # Temperature slider
# temperature = st.slider("Set Model Temperature", 0.0, 1.0, 0.3, 0.1)  # Default 0.3, step 0.1

# # Check if Groq API key is provided
# if api_key:
#     llm = ChatGroq(groq_api_key=api_key, model_name="Deepseek-R1-Distill-Llama-70b", temperature=temperature)

#     # Manage chat history
#     if "store" not in st.session_state:
#         st.session_state.store = {}

#     session_id = st.text_input("Session ID", value="default_session")

#     # File uploader for PDF
#     uploaded_files = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)
#     if uploaded_files:
#         documents = []
#         for uploaded_file in uploaded_files:
#             with open(f"./temp_{uploaded_file.name}", "wb") as temp_pdf:
#                 temp_pdf.write(uploaded_file.read())
            
#             loader = PyPDFLoader(f"./temp_{uploaded_file.name}")
#             docs = loader.load()
#             documents.extend(docs)

#         # Split documents into chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         chunks = text_splitter.split_documents(documents)

#         # Create FAISS vectorstore
#         vectorstore = FAISS.from_documents(chunks, embeddings)

#         # Create retriever
#         retriever = vectorstore.as_retriever()

#         # History-aware retriever
#         contextualize_q_system_prompt = (
#             "Given a chat history and the latest user question "
#             "which might reference context in the chat history, "
#             "formulate a standalone question that can be understood "
#             "without the chat history. Do NOT answer the question, "
#             "just reformulate it."
#         )
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

#         # QA system
#         qa_system_prompt = (
#             "You are an assistant for answering questions based on documents. "
#             "Use the provided context to answer the question exactly as written, "
#             "without summarizing. If you don't know the answer, say you don't know.\n\n{context}"
#         )
#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", qa_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
#         rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#         # Session history
#         def get_session_history(session: str) -> ChatMessageHistory:
#             if session_id not in st.session_state.store:
#                 st.session_state.store[session_id] = ChatMessageHistory()
#             return st.session_state.store[session_id]

#         conversational_rag_chain = RunnableWithMessageHistory(
#             rag_chain,
#             get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#             output_messages_key="answer",
#         )

#         # User interaction
#         user_input = st.text_input("Ask a question about your PDF:")
#         if user_input:
#             session_history = get_session_history(session_id)
#             response = conversational_rag_chain.invoke(
#                 {"input": user_input},
#                 config={"configurable": {"session_id": session_id}},
#             )
#             st.write("Assistant:", response["answer"])
#             st.write("Chat History:", session_history.messages)
# else:
#     st.warning("Please provide your Groq API Key.")


import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit app
st.title("Chat with Your PDF 📄")
st.write("Upload a PDF file and chat with its content using Groq!")

# Input the Groq API Key
#api_key = st.text_input("Enter your Groq API key:", type="password", value=os.getenv("GROQ_API_KEY"))
api_key = st.text_input("Enter your Groq API key:", type="password")
# Temperature slider
temperature = st.slider("Set Model Temperature", 0.0, 1.0, 0.3, 0.1)  # Default 0.3, step 0.1

# Check if Groq API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="Deepseek-R1-Distill-Llama-70b", temperature=temperature)

    # Manage chat history
    if "store" not in st.session_state:
        st.session_state.store = {}

    session_id = st.text_input("Session ID", value="default_session")

    # File uploader for PDF
    uploaded_files = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with open(f"./temp_{uploaded_file.name}", "wb") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
            
            loader = PyPDFLoader(f"./temp_{uploaded_file.name}")
            docs = loader.load()
            documents.extend(docs)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Create retriever
        retriever = vectorstore.as_retriever()

        # History-aware retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA system with strict document-based responses
        qa_system_prompt = (
            "You are an AI assistant that strictly answers questions from the uploaded document "
            "If the user input in irrelevent to the document or if the context does not contain relevant information, respond with: " 
            "'I can only answer questions related to the uploaded document.'\n\n"
            "Context:\n{context}"
        )
        


        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Modify retrieval chain to ensure document-based responses
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Session history
        def get_session_history(session: str) -> ChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Function to validate input and ensure document relevance
        def validate_and_answer(user_input, session_id):
            retrieved_docs = retriever.get_relevant_documents(user_input)

            if not retrieved_docs:  # If no relevant documents are retrieved
                return "I can only answer questions related to the uploaded document."

            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            return response["answer"]

        # User interaction
        user_input = st.text_input("Ask a question about your PDF:")
        if user_input:
            answer = validate_and_answer(user_input, session_id)
            st.write("Assistant:", answer)
else:
    st.warning("Please provide your Groq API Key.")



