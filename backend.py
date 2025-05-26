# from fastapi import FastAPI
# from pydantic import BaseModel
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from fastapi.middleware.cors import CORSMiddleware
#
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.llms import HuggingFacePipeline

from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Updated imports from langchain_community
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
app = FastAPI()
qa_chain = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    global qa_chain
    txt_path = "content.txt"
    # documents = TextLoader(txt_path).load()
    documents = TextLoader(txt_path, encoding='utf-8').load()
    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Lightweight
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Use a lightweight model (distilgpt2)
    model_id = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.post("/ask")
async def ask_legal_question(request: QueryRequest):
    response = qa_chain.run(request.query)
    return {"answer": response}
