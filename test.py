# test_pipeline.py
from src.ingestion_agent import IngestionAgent
from src.rag_agent import RAGAgent
from src.config import TARGET_QUESTION

def main():
    # Step 1: Ingestion
    #ingestor = IngestionAgent()
    #ingestor.ingest()
    #print("Stats:", ingestor.stats())

    # Step 2: Query
    rag = RAGAgent()
    response = rag.query(TARGET_QUESTION)
    print("\n=== RAG RESPONSE ===")
    print(response)

if __name__ == "__main__":
    main()
