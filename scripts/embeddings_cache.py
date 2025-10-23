import sqlite3
import hashlib
import pickle
import os

CACHE_DIR = 'cache'
CACHE_DB = os.path.join(CACHE_DIR, 'cache.db')

os.makedirs(CACHE_DIR, exist_ok=True)

conn = sqlite3.connect(CACHE_DB)
print(f"✓ Connected to SQLite database at {CACHE_DB}")

conn.execute(
    """
    CREATE TABLE IF NOT EXISTS embeddings_cache (
    hash TEXT PRIMARY KEY,
    embedding BLOB
)
    """
)

conn.commit()

def get_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cached_embeddings(text: str):
    h = get_hash(text)
    cursor = conn.execute("SELECT embedding FROM embeddings_cache WHERE hash=?", (h,)) 
    row = cursor.fetchone()
    if row:
        return pickle.loads(row[0])
    return None

def save_embeddings(text:str , embeddings):
    h = get_hash(text)
    conn.execute("INSERT OR REPLACE INTO embeddings_cache (hash, embedding) VALUES (?, ?)", (h, pickle.dumps(embeddings)))
    conn.commit()
