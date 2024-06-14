## Question Answering Bot created using Retrieval Augmented Generation


### Step 1: Create a RAG model based on a given PDF.
### Step 2: Measure the accuracy of model using evaluation metrics and try to improve this to a reasonable accuracy.

# load_data():
The `load_data` function initializes an environment variable with an OpenAI API key, loads a PDF file using `PyPDFLoader`, splits its contents into chunks of text, and processes these chunks into documents. It then computes embeddings for each document using `OpenAIEmbeddings` and creates a retriever database (`Chroma`) from these embeddings. Finally, it returns the retriever database, which can be used for retrieval augmented generation tasks. The function facilitates efficient handling of PDF content, text segmentation, embedding computation, and database creation tailored for retrieval-based applications.
