# hybrid_chat.py
import json
import time
from typing import List
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import config
from embeddings_cache import get_cached_embeddings, save_embeddings 

# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# Flag to enable/disable Neo4j (set to False if Neo4j is unavailable)
USE_NEO4J = True

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(
    api_key=config.GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
try:
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating managed index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    print(f"✓ Connected to Pinecone index: {INDEX_NAME}")
except Exception as e:
    print(f"✗ Failed to connect to Pinecone: {e}")
    exit(1)

# Connect to Neo4j with error handling
neo4j_driver = None
if USE_NEO4J:
    try:
        neo4j_driver = GraphDatabase.driver(
            config.NEO4J_URI, 
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        # Test connection
        neo4j_driver.verify_connectivity()
        print(f"✓ Connected to Neo4j at {config.NEO4J_URI}")
    except AuthError as e:
        print(f"⚠️  Neo4j authentication failed: {e}")
        print(f"   Username: {config.NEO4J_USER}")
        print(f"   Please check your NEO4J_PASSWORD in config.py")
        neo4j_driver = None
        USE_NEO4J = False
    except ServiceUnavailable as e:
        print(f"⚠️  Neo4j service unavailable: {e}")
        print(f"   Please start Neo4j at {config.NEO4J_URI}")
        neo4j_driver = None
        USE_NEO4J = False
    except Exception as e:
        print(f"⚠️  Failed to connect to Neo4j: {e}")
        neo4j_driver = None
        USE_NEO4J = False

if not USE_NEO4J:
    print("   → Running in Pinecone-only mode\n")

# -----------------------------
# Helper functions
# -----------------------------
def embed_text(text: str, retry_count=3) -> List[float]:
    """Get embedding for a text string with retry logic."""
    cached = get_cached_embeddings(text)
    if cached:
        print("🧠 Cache hit: embedding loaded from Redis")
        return cached
    for attempt in range(retry_count):
        try:
            # Rate limiting: wait between requests
            time.sleep(0.5)
            
            resp = client.embeddings.create(
                model=EMBED_MODEL, 
                input=[text],
                dimensions=1536
            )
            embedding = resp.data[0].embedding
            save_embeddings(text, embedding)
            return embedding
            
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                wait_time = (attempt + 1) * 10  # Exponential backoff
                print(f"⚠️  Rate limit hit. Waiting {wait_time}s... (attempt {attempt + 1}/{retry_count})")
                time.sleep(wait_time)
            else:
                print(f"✗ Embedding error: {e}")
                if attempt == retry_count - 1:
                    raise
    
    raise Exception("Failed to get embeddings after retries")

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone index using embedding."""
    try:
        vec = embed_text(query_text)
        res = index.query(
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        print(f"DEBUG: Pinecone returned {len(res['matches'])} results")
        return res["matches"]
    except Exception as e:
        print(f"✗ Pinecone query error: {e}")
        return []

def fetch_graph_context(node_ids: List[str], neighborhood_depth=1):
    """Fetch neighboring nodes from Neo4j."""
    if not USE_NEO4J or not neo4j_driver:
        return []
    
    facts = []
    try:
        with neo4j_driver.session() as session:
            for nid in node_ids:
                try:
                    q = (
                        "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                        "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                        "m.name AS name, m.type AS type, m.description AS description "
                        "LIMIT 10"
                    )
                    recs = session.run(q, nid=nid)
                    for r in recs:
                        facts.append({
                            "source": nid,
                            "rel": r["rel"],
                            "target_id": r["id"],
                            "target_name": r["name"],
                            "target_desc": (r["description"] or "")[:40],
                            "labels": r["labels"]
                        })
                except Exception as e:
                    print(f"⚠️  Error fetching graph context for node {nid}: {e}")
                    continue
        
        print(f"DEBUG: Graph returned {len(facts)} facts")
        return facts
        
    except Exception as e:
        print(f"⚠️  Graph query error: {e}")
        return []

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build a chat prompt combining vector DB matches and graph facts."""
    system = (
        "You are a helpful travel assistant for Vietnam. Use the provided semantic search results "
        "and graph facts to answer the user's query in a friendly, concise manner. "
        "Provide specific recommendations with details. "
        "When referencing places, mention their node IDs in parentheses for reference."
    )

    vec_context = []
    for m in pinecone_matches:
        meta = m.get("metadata", {})
        score = m.get("score", 0)
        snippet = f"- [{m['id']}] {meta.get('name', 'N/A')} ({meta.get('type', 'N/A')})"
        
        if meta.get("city"):
            snippet += f" in {meta.get('city')}"
        if meta.get("description"):
            snippet += f": {meta.get('description')[:15]}"
        snippet += f" [relevance: {score:.3f}]"
        
        vec_context.append(snippet)

    graph_context = []
    if graph_facts:
        for f in graph_facts:
            graph_context.append(
                f"- [{f['source']}] --{f['rel']}--> [{f['target_id']}] {f['target_name']}: {f['target_desc'][:100]}"
            )

    context_text = "Top semantic matches:\n" + "\n".join(vec_context[:10])
    
    if graph_context:
        context_text += "\n\nRelated connections (from knowledge graph):\n" + "\n".join(graph_context[:20])
    else:
        context_text += "\n\n(No graph connections available)"

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User query: {user_query}\n\n{context_text}\n\nAnswer:"}
    ]
    return prompt

def call_chat(prompt_messages, retry_count=3):
    """Call OpenAI ChatCompletion with retry logic."""
    for attempt in range(retry_count):
        try:
            time.sleep(0.5)  # Rate limiting
            
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=prompt_messages,
                max_tokens=800,
                temperature=0.3
            )
            return resp.choices[0].message.content
            
        except Exception as e:
            if "429" in str(e):
                wait_time = (attempt + 1) * 10
                print(f"⚠️  Rate limit hit. Waiting {wait_time}s... (attempt {attempt + 1}/{retry_count})")
                time.sleep(wait_time)
            else:
                print(f"✗ Chat completion error: {e}")
                if attempt == retry_count - 1:
                    return "Sorry, I encountered an error generating a response. Please try again."
    
    return "Sorry, the service is temporarily unavailable due to rate limits. Please try again in a moment."

# -----------------------------
# Interactive chat
# -----------------------------
def main():
    
    print("\n" + "="*60)
    print("🌏 Vietnam Travel Assistant (Hybrid RAG)")
    print("="*60)
    print(f"Using: Pinecone{'+ Neo4j' if USE_NEO4J else ' only'}")
    print("Type 'exit' or 'quit' to end the session\n")
    
    conversation_count = 0
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Thanks for using Vietnam Travel Assistant!")
                break
            start_time = time.time()
            conversation_count += 1
            print(f"\n[Query {conversation_count}] Processing...\n")
            
            # Step 1: Query Pinecone
            matches = pinecone_query(query, top_k=TOP_K)
            
            if not matches:
                print("⚠️  No relevant information found. Try rephrasing your question.")
                continue
            
            # Step 2: Query Neo4j graph
            match_ids = [m["id"] for m in matches]
            graph_facts = fetch_graph_context(match_ids)
            
            # Step 3: Build prompt and get answer
            prompt = build_prompt(query, matches, graph_facts)
            answer = call_chat(prompt)
            
            # Display answer
            print("="*60)
            print("📝 Answer:")
            print("="*60)
            print(answer)
            end_time = time.time()
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}\n")
            continue

if __name__ == "__main__":

    main()

