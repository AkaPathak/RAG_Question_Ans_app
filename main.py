import rag as rag
import pandas as pd
import Evaluation_metric as eval
import time
start_time = time.time()

input_file = "policy-booklet-0923.pdf"
testing_data = pd.read_csv("dataset.csv")
query_list = testing_data['Query'].tolist()
response_list = testing_data['Response'].tolist()
rag_response_list = []

retriever = rag.load_data(input_file)
for query in query_list[:5]:
    rag_response_list.append(rag.rag_answer(retriever,query))

test_size = len(rag_response_list)
for i in range(test_size):
    sentence1 = response_list[i]
    sentence2 = rag_response_list[i]
    print('s1')
    print(sentence1)
    print('s2')
    print(sentence2)
    res = eval.Cosine_Similarity(sentence1, sentence2)
    print(res) 
    r2 = eval.gpt_based_evaluation(sentence1,sentence2)
    print(r2)
    print("\n\n\n\n")

end_time = time.time()  # Capture the end time
elapsed_time = end_time - start_time 
# print(rag_response_list)
print(elapsed_time)