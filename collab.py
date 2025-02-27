!pip install pandas #Manage Data
!pip install numpy #Numerical Computing
!pip install transformers faiss-cpu sentence-transformers
import pandas as pd #กำหนดชื่อเล่นให้มันได้ ตั้งชื่อใหม่ให้กับมัน
import numpy as np #ก่อนการใช้งานทุกครั้งต้องทำการ import libary ก่อน
import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 28],
        'City': ['New York', 'London', 'Paris']}

df = pd.DataFrame(data)
print(df)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
# Load the LLM (e.g., Flan-T5)
model_name = "google/flan-t5-base" # กำหนดชื่อโมเดลที่ต้องการใช้
tokenizer = AutoTokenizer.from_pretrained(model_name) # โหลด tokenizer สำหรับโมเดล
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # โหลดโมเดล LLM

# Load the sentence transformer (e.g., all-mpnet-base-v2)
encoder = SentenceTransformer('all-mpnet-base-v2') # โหลด sentence transformer สำหรับสร้าง embedding ของข้อความ
!pip install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup
def scrape_wikipedia_page(url):
  """Scrapes content from a Wikipedia page.

  Args:
    url: The URL of the Wikipedia page.

  Returns:
    The text content of the page.
  """
  # 1. ดาวน์โหลดเนื้อหาของหน้าเว็บ
  response = requests.get(url)
  response.raise_for_status()  # Raise an exception for bad status codes

  # 2. แปลงเนื้อหา HTML เป็น BeautifulSoup object
  soup = BeautifulSoup(response.content, "html.parser")
  # Extract text from all <p> tags (paragraph elements)
  paragraphs = soup.find_all("p")
  # 4. ดึงข้อความจาก element <p> ทั้งหมด และรวมเป็น string เดียว
  text_content = " ".join([p.get_text() for p in paragraphs])

  # 5. คืนค่า text content
  return text_content
  import requests
from bs4 import BeautifulSoup

def scrape_wikipedia_page(url):
  """Scrapes content from a Wikipedia page.

  Args:
    url: The URL of the Wikipedia page.

  Returns:
    The text content of the page.
  """
  # 1. ดาวน์โหลดเนื้อหาของหน้าเว็บ
  response = requests.get(url)
  response.raise_for_status()  # Raise an exception for bad status codes

  # 2. แปลงเนื้อหา HTML เป็น BeautifulSoup object
  soup = BeautifulSoup(response.content, "html.parser")

  # 3. Extract text from all <p> tags (paragraph elements)
  paragraphs = soup.find_all("p")

  # 4. ดึงข้อความจาก element <p> ทั้งหมด และรวมเป็น string เดียว
  text_content = " ".join([p.get_text() for p in paragraphs])

  # 5. คืนค่า text content
  return text_content

# ตัวอย่างการใช้งาน
url = "https://th.wikipedia.org/wiki/%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B9%80%E0%B8%97%E0%B8%A8%E0%B9%84%E0%B8%97%E0%B8%A2"  # URL ของหน้าวิกิพีเดียเกี่ยวกับประเทศไทย
text = scrape_wikipedia_page(url)

print(text[:500])  # พิมพ์ 500 ตัวอักษรแรกของเนื้อหา
urls = [
    "https://en.wikipedia.org/wiki/Moluccan_eclectus",
    "https://en.wikipedia.org/wiki/Sun_conure",
]

documents = []
for url in urls:
    text_content = scrape_wikipedia_page(url)
    documents.append(text_content)

print(f"Scraped {len(documents)} documents.")
print(documents[0])
!pip install PyPDF2
import PyPDF2
def extract_text_from_pdf(pdf_path):
  """Extracts text content from a PDF file.

  Args:
    pdf_path: The path to the PDF file.

  Returns:
    The text content of the PDF.
  """
  with open(pdf_path, 'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    text_content = ""
    for page_num in range(num_pages):
      page = pdf_reader.pages[page_num]
      text_content += page.extract_text()
  return text_content
  pdf_file_paths = [
    "/content/Bird in my family.pdf",
    "/content/Forpus.pdf",
    "/content/The white-bellied parrot.pdf",
]

for pdf_path in pdf_file_paths:
  pdf_text = extract_text_from_pdf(pdf_path)
  documents.append(pdf_text)

print(f"Loaded {len(documents)} PDF files.")

def extract_text_from_pdf(pdf_path):
       with open(pdf_path, 'rb') as pdf_file:
           pdf_reader = PyPDF2.PdfReader(pdf_file)
           num_pages = len(pdf_reader.pages)
           text_content = ""
           for page_num in range(num_pages):
               page = pdf_reader.pages[page_num]
               text_content += page.extract_text()
       return text_content

       import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Download necessary data for sentence tokenization
nltk.download('punkt_tab')

def preprocess_text(text):
  # Remove HTML tags (if present)
  text = re.sub('<[^<]+?>', '', text)
  # Remove extra whitespace
  text = re.sub('\s+', ' ', text).strip()
  # Split into sentences
  sentences = sent_tokenize(text)
  return sentences
  processed_documents = []
for document in documents:
    processed_document = preprocess_text(document)
    processed_documents.extend(processed_document)  # Extend the list with processed sentences
    with open('/content/knowledge_base.txt', 'w') as f:
  for document in processed_documents:
    f.write(document + '\n')

    # Assuming you have loaded your models and libraries as explained in the previous response

# Generate embeddings for the knowledge base
document_embeddings = encoder.encode(processed_documents)

# Build a search index
index = faiss.IndexFlatL2(document_embeddings.shape[1])  # Create an index using L2 distance
index.add(document_embeddings)  # Add the document embeddings to the index



relevant_documents = []
with open('/content/knowledge_base.txt', 'r') as file:
    for line in file:
        relevant_documents.append(line.strip()) # .strip() removes leading/trailing whitespace

def generate_rag_response(query):
    """
    Generates a response to a given query using a Retrieval-Augmented Generation (RAG) approach.

    Args:
      query (str): The input query for which a response is to be generated.

    Returns:
      str: The generated response based on the retrieved documents and the input query.

    Steps:
      1. Embed the query using an encoder.
      2. Search for the top 10 relevant documents using the query embedding.
      3. Retrieve the relevant documents based on the search results.
      4. Format the context by combining the retrieved documents and the query.
      5. Generate the response using a language model (LLM) based on the formatted context.
    """
    # 1. Embed the query
    query_embedding = encoder.encode(query)

    # 2. Search for relevant documents
    D, I = index.search(query_embedding.reshape(1, -1), k=10)  # Search for top 5 documents

    # 3. Retrieve the relevant documents
    retrieved_documents = [relevant_documents[i] for i in I[0]]

    # 4. Format the context for the LLM
    context = f"Context: {retrieved_documents}\n\nQuery: {query}"
    print(context)

    # 5. Generate the response
    input_ids = tokenizer(context, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

    query = "What is name of her third bird?"
response = generate_rag_response(query)
print(response)