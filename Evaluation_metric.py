from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

OpenAI_api_key = "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"

def Cosine_Similarity(sentence1, sentence2):
    model_name = 'distilbert-base-nli-stsb-mean-tokens'
    model = SentenceTransformer(model_name)

    embedding1 = model.encode([sentence1])[0]
    embedding2 = model.encode([sentence2])[0]

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]

    similarity_threshold = 0.9

    if similarity >= similarity_threshold:
        return True  
    else:
        return False  
    

def gpt_based_evaluation(sentence1,sentence2):
    os.environ["OPENAI_API_KEY"] = OpenAI_api_key
    template = """
    There are two sentences s1 {sentence1} and s2 {sentence2}. Compare the meanings of these two. Return "True" if they have similar meanings. 
    Otherwise return "False".
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4", temperature=0)
    chain = (
        prompt
        | model
        | StrOutputParser()
    )
    answer = chain.invoke({"sentence1":sentence1,"sentence2":sentence2})
    return answer