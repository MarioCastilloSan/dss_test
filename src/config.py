
import os

import json
from typing import Dict, Any

#route pather
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
GEOGRAPHIC_DATA_PATH = os.path.join(BASE_DIR, "data", "geographic_data.json")

# Model config for the mistral 7b local 
MODEL_CONFIG = {
    "temperature": 0.1,
    "n_ctx": 4096,
    "n_threads": 3,
    "n_batch": 512,
    "verbose": False,
    "f16_kv": True,
    "n_gpu_layers": 0,
    "use_mlock": False
}





def load_geographic_data()->Dict[str, Any]:
    """Load geographic data from a JSON file.

    Returns:
        Dict[str, Any]: The geographic data.
    """
    try:
        with open(GEOGRAPHIC_DATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "regiones": {},
            "regiones_nombres": []
        }




# Geo data
GEOGRAPHIC_DATA = load_geographic_data()
REGIONES = GEOGRAPHIC_DATA["regiones_nombres"]



# text processing for mistral7b
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# vectorial db
COLLECTION_NAME = "chinchilla_docs"
METADATA_FIELDS = ["source", "page", "region"]

# test query
TARGET_QUESTION = "¿En qué proyectos fue relevante la chinchilla chinchilla?"

