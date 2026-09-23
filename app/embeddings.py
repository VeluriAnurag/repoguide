"""Embedding model interface.

Implement the BGE model here after the ingestion/chunking tests pass.
"""


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None

    def load(self):
        # TODO: initialize SentenceTransformer.
        raise NotImplementedError

    def encode(self, texts: list[str]):
        # TODO: return normalized embeddings.
        raise NotImplementedError
