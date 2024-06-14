
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import pandas as pd

#OpenAI Key
OpenAI_api_key = "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"


input_file = "policy-booklet-0923.pdf"
os.environ["OPENAI_API_KEY"] = OpenAI_api_key
loader = PyPDFLoader(input_file)
data = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

template = """
Using this pdf {pdf} create 50 question:answer pairs from the data available here. The data is about car insurance policy. 
The questions should be from the perspective of a 
customer who is enquiring about their car insurance. Create the unique question-answer pairs and print them one by one. You can also extract existing
question-asnwer pairs in the document. Make the answers detailed and true to the original data.
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0)
chain = (
    prompt
    | model
    | StrOutputParser()
)
answer = chain.invoke({"pdf":retriever})
print(answer)
print("\n\n\n")
q = []
ans = []
for a in answer.split('\n\n'):
    q.append(a.split('?')[0].split(':')[1]+'?')
    ans.append(a.split('?')[1].split(':')[1])

data = {
    'Query': q,
    'Response': ans
}
df = pd.DataFrame(data)
df.to_csv("dataset.csv")