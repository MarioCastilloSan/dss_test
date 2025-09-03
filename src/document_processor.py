# src/document_processor.py
import os
import shutil
import re
from typing import Dict, List, Any, Union
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastembed import TextEmbedding
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, METADATA_FIELDS, REGIONES

class DocumentProcessor:
    """Document Handler class for processing and embedding documents.
    """
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.unprocessed_dir = os.path.join(docs_dir, "unprocessed")
        self.processed_dir = os.path.join(docs_dir, "processed")

        os.makedirs(self.unprocessed_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.loaders: Dict[str, Any] = {
            ".pdf": PyPDFLoader,
            ".md": self._load_markdown,
        }
        
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )


    
    @property
    def embeddings(self) -> TextEmbedding:
        """Get the embedding model.

        Returns:
            TextEmbedding: The embedding model instance.
        """
        return self.embedding_model
    
    def has_unprocessed_documents(self) -> bool:
        """Check if there are unprocessed documents.

        Returns:
            bool: True if there are unprocessed documents, False otherwise.
        """
        return bool(os.listdir(self.unprocessed_dir))
    
    def load_documents(self) -> List[Document]:
        """Load documents from the unprocessed directory.
            Select and load documents based on their file extension.

        Returns:
            List[Document]: A list of loaded documents.
        """
        if not self.has_unprocessed_documents():
            return []
        
        documents = []
        processed_files = []
        
        for filename in os.listdir(self.unprocessed_dir):
            file_path = os.path.join(self.unprocessed_dir, filename)
            
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                
                if file_ext in self.loaders:
                    try:
                        if file_ext == ".md":
                            docs = self.loaders[file_ext](file_path, filename)
                        else:
                            loader = self.loaders[file_ext](file_path)
                            docs = loader.load()
                        
                        docs = self._ensure_documents(docs, filename)
                        
                        for doc in docs:
                            if "page" not in doc.metadata:
                                doc.metadata["page"] = doc.metadata.get("chunk_index", 0)
                            
                            doc.metadata.update({
                                "source": filename,
                                "region": self._extract_region(doc.page_content)
                            })
                        
                        documents.extend(docs)
                        processed_files.append(filename)
                    except Exception as e:
                        if file_ext == ".md":
                            print(f"Error MD {filename}: {str(e)}")
                        continue
        
        self._move_processed_files(processed_files)
        return documents
    
    def _load_markdown(self, file_path: str, filename: str) -> List[Document]:
        """Load a Markdown document.

        Args:
            file_path (str): The path to the Markdown file.
            filename (str): The name of the Markdown file.

        Returns:
            List[Document]: A list of Document objects.
        """
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            
            if not docs or not docs[0].page_content.strip():
                return self._load_markdown_fallback(file_path, filename)
                
            return docs
        except Exception:
            return self._load_markdown_fallback(file_path, filename)
    
    def _load_markdown_fallback(self, file_path: str, filename: str) -> List[Document]:
        """Fallback method to load a Markdown document.
        In case the primary loading method fails, this method attempts to read the file directly.

        Args:
            file_path (str): The path to the Markdown file.
            filename (str): The name of the Markdown file.

        Returns:
            List[Document]: A list of Document objects.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
 
            page_pattern = r'---Page (\d+)---'
            sections = re.split(page_pattern, content)
            
            docs = []
            current_page = 1
            if sections and sections[0].strip():
                docs.append(Document(
                    page_content=sections[0].strip(),
                    metadata={
                        "source": filename,
                        "page": current_page,
                        "chunk_index": 0
                    }
                ))
                current_page += 1
            
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    page_num = sections[i]
                    page_content = sections[i + 1].strip()
                    
                    if page_content:
                        docs.append(Document(
                            page_content=page_content,
                            metadata={
                                "source": filename,
                                "page": int(page_num),
                                "chunk_index": len(docs)
                            }
                        ))
            
            return docs
        except Exception as e:
            print(f"Error en fallback MD {filename}: {str(e)}")
            return []
    
    def _ensure_documents(self, raw_docs: Union[List[Document], List[str]], filename: str) -> List[Document]:
        """Ensure that all documents have the necessary metadata.

        Args:
            raw_docs (Union[List[Document], List[str]]): The raw documents to process.
            filename (str): The name of the file being processed.

        Returns:
            List[Document]: A list of Document objects with metadata.
        """
        documents = []
        
        for i, raw_doc in enumerate(raw_docs):
            if isinstance(raw_doc, Document):
                if "source" not in raw_doc.metadata:
                    raw_doc.metadata["source"] = filename
                if "chunk_index" not in raw_doc.metadata:
                    raw_doc.metadata["chunk_index"] = i
                if "page" not in raw_doc.metadata:
                    raw_doc.metadata["page"] = i
                documents.append(raw_doc)
            elif isinstance(raw_doc, str):
                doc = Document(
                    page_content=raw_doc,
                    metadata={
                        "source": filename,
                        "chunk_index": i,
                        "page": i
                    }
                )
                documents.append(doc)
            else:
                doc = Document(
                    page_content=str(raw_doc),
                    metadata={
                        "source": filename,
                        "chunk_index": i,
                        "page": i
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _move_processed_files(self, processed_files: List[str]) -> None:
        """Move processed files to the processed directory.

        Args:
            processed_files (List[str]): A list of filenames to move.
        """
        for filename in processed_files:
            src_path = os.path.join(self.unprocessed_dir, filename)
            dst_path = os.path.join(self.processed_dir, filename)
            
            try:
                shutil.move(src_path, dst_path)
            except Exception:
                continue
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks to fit within the model's context window.

        Args:
            documents (List[Document]): A list of Document objects to split.

        Returns:
            List[Document]: A list of Document objects representing the split chunks.
        """
        if not documents:
            return []
        
        valid_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                valid_docs.append(doc)
            else:
                valid_docs.append(Document(page_content=str(doc)))
        
        chunks = self.text_splitter.split_documents(valid_docs)
        return chunks
    
    def generate_embeddings(self, chunks: List[Document]) -> List[List[float]]:
        """Generate embeddings for the given document chunks.

        Args:
            chunks (List[Document]): A list of Document objects to embed.

        Returns:
            List[List[float]]: A list of embeddings, one for each document chunk.
        """
        if not chunks:
            return []
        
        texts = [chunk.page_content for chunk in chunks]
        embeddings = list(self.embedding_model.embed(texts))
        return embeddings
    
    def _extract_region(self, text: str) -> str:
        """Attempt to extract the region from the given text.

        Args:
            text (str): The text to extract the region from.

        Returns:
            str: The extracted region.
        """
        text_lower = text.lower()
        for region in REGIONES:
            if region.lower() in text_lower:
                return region

        return "Unknown"

    def process_pipeline(self) -> tuple[List[Document], List[List[float]]]:
        """Execute the full processing pipeline: load, split, and embed documents.

        Returns:
            tuple[List[Document], List[List[float]]]: The processed document chunks and their embeddings.
        """
     
        documents = self.load_documents()
        if not documents:
            return [], []
        
        chunks = self.split_documents(documents)
        if not chunks:
            return documents, []
        
        embeddings = self.generate_embeddings(chunks)
        return chunks, embeddings