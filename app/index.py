"""FAISS index interface."""


class RepositoryIndex:
    def __init__(self):
        self.index = None
        self.metadata = []

    def build(self, embeddings, metadata):
        # TODO: create a FAISS index and add embeddings.
        raise NotImplementedError

    def search(self, query_embedding, k: int = 5):
        # TODO: return the top-k metadata records.
        raise NotImplementedError
