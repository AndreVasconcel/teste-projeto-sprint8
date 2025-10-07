import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_aws import BedrockEmbeddings, BedrockLLM

session = boto3.Session()
s3_client = session.client("s3", region_name="us-east-1")
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")

# Configurações
bucket_name = "projeto04-rag"
pdf_key = "24-acordao-embargos.pdf"

# Carregar PDF
pdf_obj = s3_client.get_object(Bucket=bucket_name, Key=pdf_key)
pdf_bytes = pdf_obj['Body'].read()

with open("/tmp/temp.pdf", "wb") as f:
    f.write(pdf_bytes)

loader = PyPDFLoader("/tmp/temp.pdf")
docs = loader.load()

# Text splitter
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)
docs_splitted = r_splitter.split_documents(docs)

# Embeddings
embedding = BedrockEmbeddings(
    client=bedrock_client,
    model_id="amazon.titan-embed-text-v1"
)

# Vector store
vector_store = Chroma.from_documents(
    documents=docs_splitted,
    embedding=embedding,
    persist_directory="vector_store/chroma/"
)

# Prompt template otimizado
system_template = """Você é um assistente jurídico especializado. 
Responda à pergunta com base EXCLUSIVAMENTE no contexto fornecido.

INSTRUÇÕES:
- Seja claro, conciso e profissional
- Estruture a resposta de forma lógica
- Use marcação **negrito** apenas para termos jurídicos importantes
- Mantenha a resposta entre 150-300 palavras
- Não invente informações fora do contexto

Contexto: {context}"""

human_template = "{question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", human_template)
])

# LLM config
llm = BedrockLLM(
    model_id="us.deepseek.r1-v1:0",
    client=bedrock_client,
    model_kwargs={
        "max_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9
    }
)

# QA Chain - CORREÇÃO AQUI: removido score_threshold
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(
        search_kwargs={
            "k": 5  # Apenas número de documentos
        }
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def fazer_consulta_juridica(qa_chain, pergunta):
    try:
        response = qa_chain.invoke({"query": pergunta})
        resposta = response["result"]
        
        # Pós-processamento simples
        resposta = resposta.strip()
        
        # Remove linhas vazias excessivas
        linhas = [linha.strip() for linha in resposta.split('\n') if linha.strip()]
        return '\n'.join(linhas)
        
    except Exception as e:
        return f"Erro ao processar a consulta: {str(e)}"