from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

import os
from dotenv import load_dotenv
load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
# GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

#extract Data from the pdf file
def load_pdf_file(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

extracted_data=load_pdf_file(data='Data/')

# print(extracted_data)

#text splitting into chunks
def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
# print("length of the chunk is ",len(text_chunks))
# print(text_chunks)

#embedding models to convert to embeddings from the huggingface note the dimension of the vector model need in pinnecone
#384 dimension vector
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embeddings=download_hugging_face_embeddings()
# print(embeddings)


#to check the embedding working or not

# query_result=embeddings.embed_query("hello world")
# print("length",query_result,len(query_result))

#create an index in pinecone database
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "healthguru"
required_dimension = 384  # Dimension for sentence-transformers/all-MiniLM-L6-v2

# Check if index exists and has correct dimension
index_exists = index_name in pc.list_indexes().names()

if index_exists:
    # Get index info to check dimension
    index_info = pc.describe_index(index_name)
    current_dimension = index_info.dimension
    
    if current_dimension != required_dimension:
        print(f"Index '{index_name}' exists with dimension {current_dimension}, but required dimension is {required_dimension}.")
        print(f"Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted. Creating new index with correct dimension...")
        index_exists = False
    else:
        print(f"Index '{index_name}' already exists with correct dimension ({required_dimension}).")

if not index_exists:
    print(f"Creating index '{index_name}' with dimension {required_dimension}...")
    pc.create_index(
        name=index_name,
        dimension=required_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    print(f"Index '{index_name}' created successfully!")
    print("Waiting for index to be ready...")
    import time
    time.sleep(5)  # Wait a few seconds for index to initialize

print(f"\nUploading {len(text_chunks)} document chunks to Pinecone...")
print("This may take a few minutes...")
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
print(f"Successfully uploaded {len(text_chunks)} chunks to Pinecone index '{index_name}'!")

result = docsearch.similarity_search("diabetes treatment", k=3)
for doc in result:
    print(doc.page_content)
