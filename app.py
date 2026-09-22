from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore

load_dotenv()


def create_rag(url):

    # document loader
    loader = WebBaseLoader(
        web_path=url
    )

    docs = loader.load()

    print("Website Loaded")
    print("Documents:", len(docs))

    # Split the data
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(docs)

    print("Chunks:", len(docs))

    # Embedding
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    vector_db = InMemoryVectorStore.from_documents(
        documents=docs,
        embedding=embeddings
    )

    print("Vector DB Created")

    return vector_db


def ask_question(vector_db, query):

    records = vector_db.similarity_search(
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
Give me the final answer for my question based only on the provided context.

Context:
{context}

Question:
{query}

If the answer is not available in the context, say:
"I could not find that information on the provided webpage."
"""
    )

    return response.content