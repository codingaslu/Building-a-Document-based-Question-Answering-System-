### 1.Installing required packages and libraries**

```python
!pip install --upgrade langchain openai  -q

import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q

!apt-get install poppler-utils
```
### 2.Loading documents

First, we need to load the documents from a directory using the DirectoryLoader from LangChain. In this example, we assume the documents are stored in a directory called 'data'.

https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/directory_loader.html

```python
from langchain.document_loaders import DirectoryLoader
directory = '/content/data'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
len(documents)
```
### 3.Splitting documents**

Now, we need to split the documents into smaller chunks for processing. We will use the RecursiveCharacterTextSplitter from LangChain, which by default tries to split on the characters ["\n\n", "\n", " ", ""].

https://python.langchain.com/en/latest/modules/indexes/text_splitters/getting_started.html

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))
```
### 4.Embedding documents with OpenAI

Once the documents are split, we need to embed them using OpenAI's language model. First, we need to install the tiktoken library.
```python
#requires for open ai embedding
!pip install tiktoken -q
```
Now, we can use the OpenAIEmbeddings class from LangChain to embed the documents.
```python
import openai
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model_name="ada")

query_result = embeddings.embed_query("Hello world")
len(query_result)
```
### 5.Vector search with Pinecone

Next, we will use Pinecone to create an index for our documents. First, we need to install the pinecone-client.

```python

!pip install pinecone-client -q

"""Then, we can initialize Pinecone and create a Pinecone index.The Pinecone.from_documents() method creates a new Pinecone vector index using docs, embeddings, and index_name arguments. Docs are a list of smaller document chunks, embeddings convert text to numerical representations, and index_name is a unique identifier for the index. The method generates embeddings, indexes them, and creates an index object that can perform similarity searches and retrieve relevant documents.
https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pinecone.html

```python
import pinecone 
from langchain.vectorstores import Pinecone
# initialize pinecone
pinecone.init(
    api_key="pinecone api key",  # find at app.pinecone.io
    environment="env"  # next to api key in console
)

index_name = "langchain-demo"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
```
### 6.Finding similar documents

Now, we can define a function to find similar documents based on a given query.

```python

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

query = "How is india's economy"
similar_docs = get_similiar_docs(query)
similar_docs
```
### 7.Question answering using LangChain and OpenAI LLM

With the necessary components in place, we can now create a question-answering system using the OpenAI class from LangChain and a pre-built question-answering chain.

```python
from langchain.llms import OpenAI

# model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
model_name = "gpt-4"
llm = OpenAI(model_name=model_name)
```
### 8.Example queries and answers
Finally, let's test our question answering system with some example queries.
https://python.langchain.com/en/latest/use_cases/question_answering.html

```python
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  # print(similar_docs)
  answer =  chain.run(input_documents=similar_docs, question=query)
  return  answer

query = "Your_query"  
get_answer(query)
```
```python
query = "Your_query"
get_answer(query)
```
