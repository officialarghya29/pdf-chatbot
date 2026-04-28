from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import shutil
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = None


@app.get("/")
def home():
    return {"message": "AI PDF Chatbot Backend Running"}


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global db

    file_path = f"temp_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    db = FAISS.from_documents(docs, embeddings)

    return {"message": "PDF processed successfully"}


@app.get("/ask/")
async def ask_question(query: str):
    global db

    if db is None:
        return {"error": "Upload PDF first"}

    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ),
        retriever=retriever
    )

    result = qa.run(query)

    return {"answer": result}
