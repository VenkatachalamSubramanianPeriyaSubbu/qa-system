# Aurora QA System

A natural language question-answering system for Aurora member data using Graph RAG (Retrieval-Augmented Generation).

## Project Structure

```
qa-system/
├── app.py                    # Main FastAPI application
├── qa_processor.py           # QA processing with Graph RAG + OpenAI
├── graph_rag_retriever.py    # Graph RAG with knowledge graph
├── data_fetcher.py           # Data fetching from Aurora API
├── data_analyzer.py          # Simple analyzer for API endpoint
├── data_analyzer.ipynb       # Jupyter notebook for detailed analysis
├── prompts.py                # Centralized AI prompts
├── config.py                 # Configuration from .env
├── models.py                 # Pydantic models
└── analysis-artifacts/       # Folder for saved graphs and reports
```

## Key Features

- **Graph RAG**: Knowledge graph-based retrieval using NetworkX for understanding entity relationships
- **Entity Extraction**: Automatically extracts people, places, dates, and topics from messages
- **Relationship Mapping**: Builds connections between entities (travels, visits, preferences, etc.)
- **Hybrid Retrieval**: Combines graph traversal with semantic similarity for accurate results
- **OpenAI Integration**: Uses GPT-4o-mini for fast, cost-effective responses
- **Rate Limit Handling**: Gracefully handles Aurora API rate limits with 5-second delays
- **Jupyter Notebook Analysis**: Run `data_analyzer.ipynb` for detailed analysis with visualizations

## API Endpoints

- `GET /` - Health check and service info
- `GET /health` - Detailed health status with cache information
- `POST /ask` - Submit natural language questions
- `GET /analyze` - Get basic data analysis insights

## Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here
AURORA_API_URL=https://november7-730026606190.europe-west1.run.app

# Optional
PORT=8000
DEBUG=true
```

**Important:**
- `OPENAI_API_KEY` is **required**
- `AURORA_API_URL` is **required**
- The application will not start without these variables
- All configuration is loaded from `.env` - no hardcoded values

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The following will be installed:
# - FastAPI + Uvicorn (API server)
# - OpenAI (LLM integration)
# - NetworkX (knowledge graph)
# - Sentence-Transformers (embeddings)
# - Pandas, Matplotlib, Seaborn (data analysis)
```

## Running the Application

```bash
# Run the server
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Note:** On first startup:
1. The embedding model (`all-MiniLM-L6-v2`) will be downloaded automatically (~80MB)
2. Messages are fetched from Aurora API with 5-second delays between requests
3. A knowledge graph is built from all fetched messages

## How Graph RAG Works

The system uses **Graph RAG** - an advanced approach that builds a knowledge graph:

### 1. Entity Extraction
- **People**: Message authors and mentioned individuals
- **Places**: Locations (London, Paris, Tokyo, etc.)
- **Dates**: Temporal references
- **Topics**: Extracted from message content

### 2. Relationship Building
- `TRAVELS_TO`: Travel intentions
- `VISITS`: Visit plans
- `PREFERS`: Preferences
- `LIKES`: Interests
- `WANTS`: Desires
- `MENTIONS`: Entity references

### 3. Knowledge Graph Structure
```
Person Node → SENT → Message Node → MENTIONS → Entity Node
                                                    ↓
                                            TRAVELS_TO
                                                    ↓
                                            Location Node
```

### 4. Hybrid Retrieval Process
When a question arrives:
1. Extract entities from the question
2. Find related entities via graph traversal
3. Retrieve candidate messages connected to relevant entities
4. Rank by semantic similarity (embeddings)
5. Boost scores based on graph connections
6. Send top 10 messages to OpenAI GPT-4o-mini

### Benefits Over Vector RAG:
- **Better Context Understanding**: Knows that "Paris trip" relates to "France travel"
- **Relationship Awareness**: Finds indirect connections through graph
- **Entity-Centric**: Groups information by people, places, topics
- **More Accurate**: Combines structured (graph) and unstructured (semantic) search
- **Explainable**: Can trace why messages were retrieved via graph paths

## Data Analysis

For detailed data analysis with visualizations:

```bash
# Start Jupyter notebook
jupyter notebook

# Open data_analyzer.ipynb and run all cells
```

The notebook generates:
- Messages per member chart
- Message length distribution
- Data quality metrics
- Summary report

All visualizations are saved to `analysis-artifacts/` folder as PNG files.

## Data Insights: Anomalies and Inconsistencies

Based on comprehensive analysis of the Aurora member dataset (3,349 messages from 10 members), several key insights and anomalies were discovered:

### Overall Data Quality: 99.81%

The dataset demonstrates high quality with complete field coverage, but several noteworthy issues were identified:

### 1. **Message Length Outliers** (25 detected)
- **Finding**: 0.75% of messages (25 out of 3,349) have unusual lengths that fall outside the interquartile range
- **Distribution**: Most messages average 68 characters, but outliers range from 9 to 105 characters
- **Examples**:
  - Extremely short messages: "Yes" (9 chars), "I'd love to!" (11 chars)
  - Unusually long messages: Up to 105 characters (system may truncate longer content)
- **Impact**: These outliers could indicate data truncation, incomplete message fetching, or genuine variation in communication style
- **Affected Members**: All 10 members have at least one outlier, suggesting this is a system-wide pattern rather than individual behavior

### 2. **Aurora API Inconsistencies**
During data collection, the Aurora API exhibited several problematic behaviors:
- **403 Forbidden**: Intermittent authentication/authorization failures mid-fetch
- **404 Not Found**: Some pagination offsets return "not found" despite being within range
- **402 Payment Required**: Unexpected payment errors during normal operation
- **400 Bad Request**: Malformed request errors at specific skip offsets
- **Pattern**: These errors appear at specific offset ranges (e.g., skip=100, 300, 800, 900, 1300)

**Implications**:
- The API may have rate limiting that's not documented
- Pagination boundaries might be inconsistent
- Some data pages might be temporarily unavailable
- This required implementing robust retry logic with exponential backoff

### 3. **Member Activity Distribution**
- **Finding**: Fairly balanced participation across all members
- **Range**: 288-365 messages per member (77 message difference)
- **Standard Deviation**: ~25 messages from mean (334.9 messages/member)
- **Assessment**: No inactive or suspiciously hyperactive members detected

### 4. **Data Completeness**
✅ **No issues found:**
- 0 messages with empty content
- 0 duplicate messages
- 100% complete fields (id, member_name, content, timestamp, user_id)
- All messages successfully linked to valid members

### 5. **Content Consistency**
- **Average message length**: 68 characters (reasonable for chat-like communication)
- **Length distribution**: Normal distribution with slight right skew
- **Character encoding**: UTF-8, no encoding errors detected
- **Timestamp validity**: All timestamps valid and in chronological order

### Summary
The dataset is production-ready with minor caveats:
- ✅ Excellent data completeness and no duplicates
- ✅ Consistent member participation
- ✅ Valid timestamps and field data
- ⚠️  Aurora API reliability issues require retry logic
- ⚠️  Message length outliers may indicate content truncation
- ⚠️  Some data pages occasionally unreachable

**Recommendation**: The current implementation handles these issues gracefully through:
1. Retry logic with exponential backoff
2. 5-second delays between requests to avoid rate limits
3. Partial data acceptance when some pages fail
4. Caching to minimize API calls

For detailed statistics and visualizations, see the `analysis-artifacts/` folder or run `data_analyzer.ipynb`.

## API Usage Examples

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is planning to travel to Paris?"}'
```

Response:
```json
{
  "answer": "Based on the member messages, Alice mentioned planning a trip to Paris in spring..."
}
```

### Health Check

```bash
curl "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy",
  "cache_status": "loaded",
  "cache_timestamp": "2025-11-14T10:30:00.123456"
}
```

### Data Analysis

```bash
curl "http://localhost:8000/analyze"
```

## Architecture Overview

```
User Question
     ↓
FastAPI (/ask endpoint)
     ↓
QA Processor
     ↓
Graph RAG Retriever
     ↓
1. Extract entities from question
2. Traverse knowledge graph
3. Find connected messages
4. Rank by semantic similarity
5. Return top 10 messages
     ↓
OpenAI GPT-4o-mini
     ↓
Generated Answer
     ↓
User Response
```

## Design Notes: Alternative Approaches Considered

While building this QA system, I explored several different approaches before settling on the current Graph RAG implementation. Here's what I considered and why I made the choices I did:

### 1. Naive Approach: Everything in the Prompt

**What it is:** Just dump all the messages directly into the LLM prompt and let it figure things out.

**How it would work:**
- Fetch all messages from Aurora API
- Concatenate everything into one massive string
- Send it all to OpenAI with the user's question
- Hope the LLM can find the relevant info

**Why I considered it:**
- Dead simple to implement
- No fancy retrieval logic needed
- Works great for small datasets

**Why I didn't go with it:**
- **Token limits**: OpenAI has a context window limit. With hundreds of messages, you'd hit that wall fast
- **Cost**: Every API call would be expensive since you're sending thousands of tokens
- **Performance**: The more context you give an LLM, the slower and less focused it becomes
- **Accuracy**: LLMs can miss relevant info when buried in massive context (the "needle in a haystack" problem)

**Verdict:** Great for prototyping with 10-20 messages, terrible for production with 3600+ messages.

### 2. Vector RAG with FAISS

**What it is:** Convert all messages to embeddings (numerical vectors) and use similarity search to find relevant ones.

**How it works:**
- Use sentence-transformers to convert each message to a 384-dimensional vector
- Store all vectors in a FAISS index (Facebook's similarity search library)
- When a question arrives, convert it to a vector too
- Find the K most similar message vectors using cosine similarity
- Send only those messages to the LLM

**Why I considered it:**
- Industry standard approach for RAG
- FAISS is lightning fast (can search millions of vectors in milliseconds)
- Semantic search works really well for finding similar content
- Well-documented with tons of examples

**Implementation details:**
- Used `all-MiniLM-L6-v2` model (384 dimensions, 80MB)
- L2 distance for similarity
- Retrieved top 10 messages per query
- Reduced token usage by ~98%

**Why I initially went with it:**
- Proven technology
- Simple to implement
- Works great for semantic similarity

**Limitations I discovered:**
- **No relationship understanding**: If someone asks "Who's traveling with Alice?", vector search might find Alice's messages but miss Bob's message that says "I'm joining Alice"
- **Context-blind**: Doesn't know that "Paris" and "France" are related, or that "trip" and "vacation" mean similar things
- **Entity-agnostic**: Treats people, places, and events the same way
- **No multi-hop reasoning**: Can't connect "Alice → Paris" and "Paris → France" to answer "Who's going to France?"

**Verdict:** Solid approach and works well, but leaves some accuracy on the table.

### 3. Graph RAG with Neo4j

**What it is:** Build a full-fledged knowledge graph database with nodes and relationships, then query it.

**How it would work:**
- Set up Neo4j database (graph database)
- Extract entities (people, places, dates) from messages
- Create nodes for each entity
- Create relationships between entities (TRAVELS_TO, MENTIONS, etc.)
- Use Cypher queries to traverse the graph
- Combine graph results with semantic search

**Why I was excited about it:**
- **Powerful queries**: Cypher language lets you write complex graph traversals
- **Real database**: Persistent storage, ACID transactions, indexing
- **Scalability**: Neo4j handles millions of nodes efficiently
- **Visualization**: Built-in tools to visualize the knowledge graph
- **Multi-hop reasoning**: Can find indirect connections naturally

**Example Cypher query:**
```cypher
MATCH (person:Person)-[:TRAVELS_TO]->(place:Location)
WHERE place.name CONTAINS 'Paris'
RETURN person.name, person.messages
```

**Why I didn't go with it:**
- **Complexity**: Requires running a separate database server
- **Deployment overhead**: Need to host Neo4j alongside the API
- **Overkill for this scale**: 600 messages don't need a full graph database
- **Cost**: Neo4j hosting or server resources
- **Setup time**: More moving parts mean more things to configure and maintain

**Verdict:** Amazing for large-scale production systems with millions of entities, but too heavy for this project's scope.

### 4. What I Actually Built: Lightweight Graph RAG with NetworkX

**The compromise:** Get the benefits of graph-based reasoning without the complexity of a database.

**Why this approach wins:**
- **In-memory graph**: NetworkX builds the graph in Python, no external database needed
- **Fast enough**: For 600 messages, in-memory is actually faster than database queries
- **Simple deployment**: Just `pip install networkx`, no servers to manage
- **Graph reasoning**: Still get entity extraction, relationships, and graph traversal
- **Hybrid retrieval**: Combine graph connections with semantic similarity scoring
- **Easy to debug**: The whole graph is in memory, can inspect it anytime

**How it works:**
1. Extract entities from each message (people, places, dates)
2. Build NetworkX directed graph with nodes and edges
3. When a question comes in, extract entities from the question
4. Traverse the graph to find connected messages
5. Score candidates using semantic similarity
6. Boost scores based on graph connectivity
7. Return top 10 most relevant messages

**Trade-offs made:**
- Not as powerful as Neo4j's Cypher queries (but we don't need them)
- Graph is rebuilt on cache refresh (but that's fine for this scale)
- No persistent storage (but we're fetching from API anyway)
- Simpler entity extraction (but good enough for the use case)

**Why this is the sweet spot:**
- Gets 80% of the benefit with 20% of the complexity
- No additional infrastructure needed
- Fast and accurate for this dataset size
- Easy to understand and modify
- Production-ready without being over-engineered

### Bonus: Other Things I Considered

**Hybrid with both FAISS and Graph:**
Keep FAISS for pure semantic search, use graph for entity-based queries. Decided against it because it adds complexity without clear benefits for this scale.

**Fine-tuning a small LLM:**
Train a custom model on the Aurora data. Rejected because:
- Need way more data for fine-tuning
- Expensive and time-consuming
- Pre-trained models already work great
- Would need retraining as data changes

**Elasticsearch:**
Use Elasticsearch for full-text search + semantic search. Good option but:
- Another service to run
- Similar to Neo4j complexity problem
- NetworkX approach is simpler and good enough

## Rate Limiting and Retry Logic

The system is resilient to API failures:
- 5-second delay between normal requests
- 10-second pause before retrying failed requests
- Up to 3 retry attempts per failed page
- Gracefully handles 401, 402, 403, 404, 429 errors
- Returns partial data if some pages succeed
- Caches data for 5 minutes to reduce API calls

## Development

### Project Stack
- **Framework**: FastAPI
- **LLM**: OpenAI GPT-4o-mini
- **Graph Database**: NetworkX
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Data Analysis**: Pandas, Matplotlib, Seaborn
- **Async HTTP**: aiohttp

### Key Design Decisions
1. **Graph RAG over Vector RAG**: Better relationship understanding
2. **OpenAI over Gemini**: More reliable, consistent quota
3. **In-Memory Graph**: Fast access, rebuilt on cache refresh
4. **5-Second Request Delay**: Avoids Aurora API rate limits
5. **SSL Disabled**: For Aurora API development environment

## Troubleshooting

### OpenAI API Errors
- Check `OPENAI_API_KEY` in `.env`
- Verify API key is valid and has credits
- Check quota limits on OpenAI dashboard

### Aurora API Errors
- Ensure `AURORA_API_URL` is correct
- Rate limits: System automatically handles with partial data
- Connection issues: Check network and API availability

### Import Errors
```bash
# NumPy compatibility issue with sentence-transformers
pip install 'numpy<2.0.0'

# Reinstall dependencies
pip install -r requirements.txt
```

## Docker Deployment

### Building the Docker Image

```bash
docker build -t aurora-qa-system .
```

### Running Locally with Docker

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-key \
  -e AURORA_API_URL=https://november7-730026606190.europe-west1.run.app \
  aurora-qa-system
```

### Deploying to Google Cloud Run

1. **Set up Google Cloud CLI:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. **Configure Docker for GCR:**
```bash
gcloud auth configure-docker
```

3. **Build and tag the image:**
```bash
PROJECT_ID="your-gcp-project-id"
IMAGE_NAME="gcr.io/${PROJECT_ID}/aurora-qa-system"
docker build -t ${IMAGE_NAME}:latest .
```

4. **Push to Google Container Registry:**
```bash
docker push ${IMAGE_NAME}:latest
```

5. **Deploy to Cloud Run:**
```bash
gcloud run deploy aurora-qa-system \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=your-key,AURORA_API_URL=https://november7-730026606190.europe-west1.run.app" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300
```

**Or use the deployment script:**
```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export AURORA_API_URL="https://november7-730026606190.europe-west1.run.app"

# Edit deploy.sh with your PROJECT_ID
nano deploy.sh

# Make executable and run
chmod +x deploy.sh
./deploy.sh
```

### Cloud Run Configuration

- **Memory**: 1GB (enough for the embedding model)
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds (for initial data loading)
- **Min Instances**: 0 (scales to zero when not in use)
- **Max Instances**: 10 (adjust based on traffic)

### Environment Variables for Cloud Run

Set these in the Cloud Run console or via `gcloud run deploy`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `AURORA_API_URL`: Aurora API endpoint
- `PORT`: Automatically set by Cloud Run (default 8080)

### Cost Optimization

Cloud Run pricing:
- Scales to zero when not in use
- Only pay for actual request time
- 1GB memory + 1 CPU is sufficient
- Embedding model (~80MB) loads on first request

## License

See LICENSE file for details.
