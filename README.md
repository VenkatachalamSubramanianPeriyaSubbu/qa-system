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

## Rate Limiting

The system handles Aurora API rate limits:
- 5-second delay between requests
- Gracefully handles 401, 402, 403, 404, 429 errors
- Uses partial data if rate limit is reached
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

## License

See LICENSE file for details.
