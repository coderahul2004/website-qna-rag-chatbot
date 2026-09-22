import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

load_dotenv()

st.subheader("Customer Support QnA Chatbot")

if "web_loaded" not in st.session_state:
    st.session_state.web_loaded = False

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


def process_urls(urls):
    alldocs = []

    for url in urls:
        loader = WebBaseLoader(web_path=url)
        docs = loader.load()
        alldocs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(alldocs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    st.session_state.vector_db = vector_db
    st.session_state.web_loaded = True


if not st.session_state.web_loaded:

    urls = st.text_area(
        "Enter website URL(s):"
    )

    if urls:
        process_urls(urls.split())

        st.success("URL Processed Successfully!")

        st.rerun()


if st.session_state.web_loaded and st.session_state.vector_db:

    query = st.chat_input("Ask Anything")

    if query:

        st.chat_message("user").markdown(query)

        records = st.session_state.vector_db.similarity_search(
            query=query,
            k=6
        )

        context = ""

        for chunk in records:
            context += chunk.page_content + "\n\n"

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash"
        )

        response = llm.invoke(
            f"""
            Answer the question using only the provided context.

            Context:
            {context}

            Question:
            {query}
            """
        )

        st.chat_message("ai").markdown(response.content)