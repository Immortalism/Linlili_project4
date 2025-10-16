from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain


url = "https://api.aigc369.com/v1"

import os
# 设置代理（替换为你的实际代理地址和端口）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"  # 示例代理地址
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"  # 示例代理地址

def qa_agent(api_key,memory,uploaded_file,question):
    model = ChatOpenAI(model="gpt-3.5-turbo",api_key=api_key,base_url=url)

    file_content = uploaded_file.read()
    temp_file_path = "temp1.pdf"
    with open(temp_file_path,"wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    embeddings_model = OpenAIEmbeddings(api_key=api_key,base_url=url)
    db = FAISS.from_documents(texts, embeddings_model)

    retriever = db.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(
        llm = model,
        retriever = retriever,
        memory = memory
        )
    
    response = qa.invoke({"chat_history":memory,"question": question})
    return response







