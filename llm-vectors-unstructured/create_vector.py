import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv()

from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

print("URI:", uri)
print("Username:", username)
print("Password configured:", bool(password))

with GraphDatabase.driver(
    uri,
    auth=(username, password),
    connection_timeout=10
) as driver:
    driver.verify_connectivity()
    print("Successfully connected to Neo4j!")


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "asciidoc"

print(f"Loading documents from: {DATA_DIR}")

loader = DirectoryLoader(
    str(DATA_DIR),
    glob="**/*.adoc",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

print(f"Loaded {len(docs)} documents.")


# Create a text splitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 1500,
    chunk_overlap = 200
)

# Split documents into chunks
chunks = text_splitter.split_documents(docs)

# Create a Neo4j vector store
neo4j_db = Neo4jVector.from_documents(
    chunks,
    OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY')),
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    database=os.getenv('NEO4J_DATABASE', 'neo4j'),
    index_name="chunkVector",
    node_label="Chunk",
    text_node_property="text",
    embedding_node_property="embedding",
)