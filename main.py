from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    language: str = "en"

openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_cortina_index", embeddings)
llm = OpenAI(temperature=0.3)
chain = load_qa_chain(llm, chain_type="stuff")

@app.post("/query")
async def query(request: QueryRequest):
    docs = vectorstore.similarity_search(request.query, k=4)
    answer = chain.run(input_documents=docs, question=request.query)
    return {"response": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
