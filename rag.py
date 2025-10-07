import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings, BedrockLLM

session = boto3.Session()
s3_client = session.client("s3", region_name="us-east-1")
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

bucket_name = "projeto04-rag"
pdf_key = "24-acordao-embargos.pdf"

pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
pdf_bytes = pdf_obj['Body'].read()

with open("/tmp/temp.pdf", "wb") as f:
    f.write(pdf_bytes)

loader = PyPDFLoader("/tmp/temp.pdf")
docs = loader.load()

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
Você é um assistente jurídico especializado. Responda à pergunta com base EXCLUSIVAMENTE no contexto fornecido.

INSTRUÇÕES:
- Seja conciso e direto
- Estruture a resposta em parágrafos curtos
- Use **negrito** apenas para termos jurídicos importantes
- Não repita informações desnecessariamente
- Se não encontrar informação suficiente no contexto, diga que não há dados suficientes
- Mantenha a resposta entre 100-300 palavras

Contexto:
{context}

Pergunta: {question}

Resposta:"""
prompt = PromptTemplate.from_template(QUERY_PROMPT_TEMPLATE)

llm = BedrockLLM(
    model_id="us.deepseek.r1-v1:0",
    client=bedrock_client,
    model_kwargs={
        "max_tokens": 1024, 
        "temperature": 0.1,  
        "top_p": 0.9,
        "stop_sequences": ["\n\n"]  
    }
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def fazer_consulta_juridica(qa_chain, pergunta):
    response = qa_chain.invoke({"query": pergunta})
    return response["result"]
