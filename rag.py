import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings, BedrockLLM

session = boto3.Session()
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

loaders = [PyPDFLoader("24-acordao-embargos.pdf")]
docs = []
for loader in loaders:
    docs.extend(loader.load())

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,
    chunk_overlap=500,
    separators=["\n\n", "\n"]
)
docs_splitted = r_splitter.split_documents(docs)

embedding = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"
)

vector_store = Chroma.from_documents(
    documents=docs_splitted,
    embedding=embedding,
    persist_directory="vector_store/chroma/"
)

QUERY_PROMPT_TEMPLATE = """\
Responda à pergunta com base exclusivamente no contexto fornecido.
Explique de forma detalhada, trazendo a fundamentação jurídica e 
fazendo referência às passagens relevantes do texto. 

Se a resposta estiver no contexto, desenvolva com clareza e completeza. 
Não invente informações fora do contexto.

Contexto:
{context}

Pergunta: {question}

Resposta detalhada:
"""

prompt = PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)

llm = BedrockLLM(
    model_id="us.deepseek.r1-v1:0",
    client=bedrock_client,
    model_kwargs={
        "max_tokens": 2048, 
        "temperature": 0.2
        }
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 7}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def fazer_consulta_juridica(qa_chain, pergunta):
    response = qa_chain.invoke({"query": pergunta})
    return response["result"]