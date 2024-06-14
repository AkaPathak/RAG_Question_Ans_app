#Installations
# pip install --upgrade --quiet  langchain langchain-openai faiss-cpu tiktoken langchain-community pypdf os

#Imports
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import os

#OpenAI Key
OpenAI_api_key = "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"

#Get the pdf data
input_file = "policy-booklet-0923.pdf"

#Function to read it and chunk it, create vector embeddings.
def load_data(file):
    """
    Loads the pdf file to be used for the retrieval augmented generation 
    and creates a vector database using it.
    """
    os.environ["OPENAI_API_KEY"] = OpenAI_api_key
    loader = PyPDFLoader(file)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    return db.as_retriever()

#RAG 
def rag_answer(context,query):
    template = """Pretend you are the respresentative of a car insurance policy. Answer the question based only on the following context:
    {context}
    Question: {question}
    After getting the question, try to find out the meaning behind it and compare its meaning to the meanings of the data stored in the context. If a question is asking about something that is available in the context, return the answer. The answer should be the same as the one in the original data.
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = (
        {"context": context, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    answer = chain.invoke(query)

    if "I'm sorry" in answer or "provided context" in answer or "not clear" in answer:
        print("Answer not found")
        print(answer)
        answer = " NULL "
    return answer

# retriever = load_data(input_file)
# query = input("Enter your question: ")
# rag_answer(retriever,query)