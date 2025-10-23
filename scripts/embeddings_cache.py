import redis
import hashlib
import pickle
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", 48))

conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def get_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cached_embeddings(text: str):
    h = get_hash(text)
    data = conn.get(h)
    if data:
        return pickle.loads(data)
    return None

def save_embeddings(text:str , embeddings, ttl_hours=48):
    h = get_hash(text)
    data = pickle.dumps(embeddings)
    conn.setex(h, ttl_hours * 3600, data)


def get_cached_answer(query: str):
    """Retrieve a cached LLM answer if available."""
    key = f"answer:{get_hash(query)}"
    data = conn.get(key)
    if data:
        return pickle.loads(data)
    return None

def save_cached_answer(query: str, answer: str, ttl_hours: int = CACHE_TTL_HOURS):
    """Save a query → answer mapping in Redis."""
    key = f"answer:{get_hash(query)}"
    data = pickle.dumps(answer)
    conn.setex(key, ttl_hours * 3600, data)