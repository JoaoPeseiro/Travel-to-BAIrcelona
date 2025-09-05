# 🧒👶 Travel Assistant for Kids

An **AI-powered travel assistant for families with children**, built using **Python + Streamlit + LangChain + Pinecone**.  
The system provides tailored recommendations for **activities, packing lists, restaurants, and logistics** by combining **local data (YouTube + websites)** with **real-time search (DuckDuckGo)**.

---

## 📐 Project Architecture

### 🔹 1. Input Layer
- **Streamlit Interface**: user-friendly chat and button-based interaction.  
- Inputs: city, child’s age, season, free-form questions.

### 🔹 2. Data Ingestion
- **VTT transcripts** from YouTube travel videos.  
- **Web scraping** using `BeautifulSoup`.  
- **Caching** with `processed_cache.json` to avoid redundant processing.  

### 🔹 3. Vector Store
- **Embeddings** generated with `SentenceTransformer` (`paraphrase-multilingual-MiniLM-L12-v2`).  
- **Pinecone** used for indexing and semantic search.  
- Metadata stored (title, source, content snippet).  

### 🔹 4. LangChain Components
- **Retrievers**:  
  - `vectorstore.as_retriever()` for direct lookups.  
  - `MultiQueryRetriever` to expand user questions into multiple sub-queries.  
- **Chains**:  
  - `RetrievalQA` with a custom `qa_prompt` for contextual answers.  
  - `LLMChain` for query reformulation and summarization of web results.  
- **Agent** (`initialize_agent`) with tools:  
  - `Local RAG QA` → answers from Pinecone.  
  - `MultiQuery RAG QA` → expands questions and retrieves diverse perspectives.  
  - `DuckDuckGo + Learn` → fetches external info, summarizes it, and updates Pinecone.  
- **Memory**: `ConversationBufferMemory` keeps multi-turn context.  
- **Tracing**: LangSmith integration for debugging and evaluation.  

### 🔹 5. Output Layer
- Responses are displayed in **Streamlit chat format**.  
- Answers are structured in **short, clear, family-friendly lists**.  

### 🔹 6. Mermaid
flowchart TD
    A[User Input: City, Age, Season] --> B[Streamlit UI]
    B --> C[LangChain Agent]
    C --> D[Tools]
    D --> D1[Local RAG QA (Pinecone Index)]
    D --> D2[MultiQuery RAG QA]
    D --> D3[DuckDuckGo + Learn]
    C --> E[ChatOpenAI (GPT-4o)]
    E --> F[Response Generation]
    F --> B


---

## 🔬 Methodology

1. **Data Collection**  
   - Extract subtitles from YouTube travel videos (`.vtt`).  
   - Scrape relevant websites (TripAdvisor, travel blogs, TheFork, etc.).  

2. **Processing & Indexing**  
   - Clean and preprocess text.  
   - Convert into embeddings.  
   - Store in Pinecone with metadata.  

3. **Knowledge Base Construction**  
   - Bot first queries local vector database.  
   - If no answer is found → fallback to DuckDuckGo search.  
   - External results are **summarized and dynamically injected** into Pinecone.  

4. **Response Pipeline**  
   - User query → LangChain Agent → Tool selection → Context retrieval → Final response generation.  

5. **Evaluation & Iteration**  
   - Quality monitoring with LangSmith tracing.  
   - Prompt refinement to reduce hallucinations.  

---

## ⚙️ LangChain Usage (Deliverables)

- **LLM**: `ChatOpenAI` (GPT-4o) as main reasoning engine.  
- **Embeddings**: `SentenceTransformerEmbeddings` for text vectorization.  
- **Retrievers**:  
  - `MultiQueryRetriever` for query expansion.  
- **Chains**:  
  - `RetrievalQA` → contextual Q&A over Pinecone.  
  - `LLMChain` → query rewriting & web result summarization.  
- **Agent**:  
  - Built with `initialize_agent` and `AgentType.ZERO_SHOT_REACT_DESCRIPTION`.  
  - Equipped with 3 tools: Local RAG, MultiQuery RAG, DuckDuckGo Search.  
- **Memory**: `ConversationBufferMemory` for conversational context.  
- **Callbacks**: LangSmith for logging, evaluation, and debugging.  

---

## 🚀 How to Run

1. Install dependencies
pip install -r requirements.txt
2. Create .env file
OPENAI_API_KEY=sk-xxx
PINECONE_API_KEY=pc-xxx
LANGSMITH_API_KEY=ls-xxx
3. Launch Streamlit app
streamlit run Travel2.py
📌 Next Steps
 Improve summarization of DuckDuckGo results.

 Add unit tests for prepare_documents and prepare_vectorstore.

 Expand dataset with more trusted sources.

✍️ Author: João Peseiro
📅 Final Project – Travel Assistant for Kids