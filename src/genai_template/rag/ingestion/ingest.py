import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser

from genai_template.config import config


class Pipeline:
    def __init__(self):
        self.nodes = None
        self.ingested = config.DATA_DIR.exists()
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        self._pipeline = IngestionPipeline(
            transformations=[
                MarkdownNodeParser(
                    include_metadata=True,
                    include_prev_next_rel=True,
                ),
            ]
        )

    async def arun(self) -> None:
        print("[DEBUG] Checking document store...")
        if self.ingested:
            print("[DEBUG] Document is already ingested.")
            return

        try:
            documents = SimpleDirectoryReader(
                input_files=[config.DOC_DIR / "document.md"]
            )
            self.nodes = await self._pipeline.arun(
                documents=documents,
                in_place=False,
            )
            print("[DEBUG] Document successfully ingested.")
        except Exception as ex:
            print("[ERROR] Ingestion error:", ex)

    def run(self) -> None:
        asyncio.run(self.aingest())
