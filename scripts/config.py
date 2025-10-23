# config_example.py — copy to config.py and fill with real values.
# config.py
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"  # ← This is what you set in docker run
GEMINI_API_KEY = "AIzaSyAIZJOjjq87FDmW9sVoTuvPkwnmfFWtfNE" # your OpenAI API key

PINECONE_API_KEY = "pcsk_66k9oF_4WXyMk54M4VEgjXfpGxf6UBqVjie8JN9uSMb1rqJFf7evWqR3BEeZS5TSoxatBW" # your Pinecone API key
PINECONE_ENV = "us-east-1"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 1536       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
