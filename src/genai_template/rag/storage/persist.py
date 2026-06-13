from typing import List

from chromadb import PersistentClient
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from genai_template.config import config

COLLECTION_NAME = "template_app"


class PersistentStore:
    def __init__(self):
        self.index: VectorStoreIndex | None = None

    def load(self) -> None:
        if not config.DATA_DIR.exists():
            raise FileNotFoundError("[ERROR] Knowledge base not found.")

        print(
            f"[DEBUG] Loading Chroma vector store: embed_model='{config.EMBED_MODEL_NAME}'"
        )
        embed_model = FastEmbedEmbedding(model_name=config.EMBED_MODEL_NAME)
        vector_store = ChromaVectorStore(persist_dir=config.DATA_DIR)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        print("[DEBUG] Vector store successfully loaded.")

    @staticmethod
    def from_nodes(nodes: List[BaseNode]):
        if not nodes or not len(nodes):
            raise ValueError(
                "[ERROR] Must be a list of nodes: Received None or empty list."
            )

        try:
            print(
                f"[DEBUG] Initializing Chroma vector store: embed_model='{config.EMBED_MODEL_NAME}'"
            )
            embed_model = FastEmbedEmbedding(model_name=config.EMBED_MODEL_NAME)
            client = PersistentClient(path=config.DATA_DIR)
            collection = client.get_or_create_collection(name=COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            persist_store = PersistentStore()
            persist_store.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
            )

            print("[DEBUG] Vector store successfully initialized.")
            return persist_store
        except Exception as ex:
            print("[ERROR] Could not initialize vector store:", ex)
