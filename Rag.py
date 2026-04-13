# !pip install langchain chromadb sentence-transformers openai pypdf
# !pip install langchain chromadb sentence-transformers openai pypdf

from langchain_community.document_loaders import PyPDFLoader
from google.colab import files
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI



file_path = "/content/Wireshark User's Guide.pdf" # This is the path of the pdf
document = PyPDFLoader(file_path).load()

#The part of create the chunks
spliter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = spliter.split_documents(document)

#Save the chunk in the vector DB
embedding = HuggingFaceEmbeddings()
vector_db = Chroma.from_documents(docs, embedding)

def retrive(query):
  return vector_db.similarity_search(query,k=3)


client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1",
                
)

def Final_answer(query):
  docs = retrive(query)
  context = "\n".join([doc.page_content for doc in docs])
  prompt = f"""
  Answer using the context below.
  {context}
  The question is:
  {query}
  """

  response = client.chat.completions.create(
      model = "openai/gpt-oss-20b",
      messages=[
          {
              "role": "system",
              "content": "your answer based on only given context"
          },
          {
              "role": "user",
              "content": prompt
          }

      ]
  )
  return response.choices[0].message.content

query = input("ask your questions: ")
ans = Final_answer(query)
print(ans)