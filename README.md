## Question Answering Bot created using Retrieval Augmented Generation


### Step 1: Create a RAG model based on a given PDF.
### Step 2: Measure the accuracy of model using evaluation metrics and try to improve this to a reasonable accuracy.

# load_data():
The `load_data` function initializes an environment variable with an OpenAI API key, loads a PDF file using `PyPDFLoader`, splits its contents into chunks of text, and processes these chunks into documents. It then computes embeddings for each document using `OpenAIEmbeddings` and creates a retriever database (`Chroma`) from these embeddings. Finally, it returns the retriever database, which can be used for retrieval augmented generation tasks. The function facilitates efficient handling of PDF content, text segmentation, embedding computation, and database creation tailored for retrieval-based applications.

# rag_answer():
The `rag_answer` function employs the Retrieval-Augmented Generation (RAG) approach to respond to a query based on a given context. It defines a template where the context and query are combined, simulating a scenario where the model acts as a car insurance representative responding to inquiries. It uses a ChatGPT model (specifically GPT-4) to generate answers by processing the context and query through a defined pipeline: context and question inputs, template-based prompt generation, model inference, and output parsing. If the model fails to generate a coherent answer (indicated by specific phrases in the response), it defaults to returning "NULL". This function facilitates automated response generation tailored to contextual queries, leveraging advanced natural language processing capabilities to assist users in understanding and utilizing insurance-related information effectively.

# cosine_similarity():
The `Cosine_Similarity` function computes the cosine similarity between two input sentences using a pre-trained SentenceTransformer model. It initializes with 'distilbert-base-nli-stsb-mean-tokens', a model trained for semantic similarity tasks. After encoding both sentences into embeddings, it calculates the cosine similarity between these embeddings. If the similarity score meets or exceeds a predefined threshold of 0.5, the function returns True, indicating the sentences are considered similar in meaning. Otherwise, it returns False. This function provides a straightforward method to quantify and determine semantic similarity between pairs of sentences, useful for tasks such as text classification, information retrieval, and semantic search applications.

# gpt_based_evaluation():
The `gpt_based_evaluation` function utilizes OpenAI's GPT-4 model to evaluate the semantic similarity between two given sentences. It sets up a template that prompts the model to compare the meanings of `sentence1` and `sentence2`. The function initializes with an API key, defines a template with placeholders for the sentences, and uses a ChatGPT model configured for inference (`ChatOpenAI(model="gpt-4", temperature=0)`). The function then processes the template through a pipeline that includes the model and an output parser (`StrOutputParser()`). It invokes the model with the provided sentences, expecting the model to respond with "True" if the sentences have similar meanings and "False" otherwise. This function leverages advanced natural language understanding capabilities of GPT-4 to perform semantic evaluations, offering a practical tool for assessing textual similarity in various applications such as content analysis and automated evaluation systems.

# The thought process behind this:

## I first created a vector database to store the chunks of the text in the pdf input. Then I created a rag pipeline that would take a query and pass it to gpt along with the vector database as a reference.

How you constructed your dataset
How and why you chose these evaluation metrics
What did you try to improve the accuracy
