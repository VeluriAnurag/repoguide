"""Embedding model interface.

An embedding model turns text into a fixed-length vector of numbers such that
texts with similar meaning get similar vectors. RepoGuide only depends on the
two public methods below, so the model can be swapped (or faked in tests).

We run the Hugging Face BGE model with ONNX Runtime rather than PyTorch.
PyTorch and FAISS each bundle their own copy of the OpenMP runtime on macOS,
and loading both into one process crashes. ONNX Runtime has no OpenMP
dependency and produces the same vectors (verified to within 1e-7).
"""

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE models are trained to expect this prefix on search queries (but not on
# the documents being searched). It noticeably improves retrieval for short
# questions.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_TOKENS = 512  # BGE cannot read more than this; longer text is cut off
BATCH_SIZE = 32


class EmbeddingModel:
    """Hugging Face BGE embeddings, run locally with ONNX Runtime."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._tokenizer = None
        self._session = None

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunks to be stored in the index. Shape: (len(texts), dim)."""
        # Every text in a batch is padded to the longest one, so mixing short
        # and long texts wastes work. Embedding them shortest-to-longest keeps
        # similar lengths together (about half the work on real repos), then
        # we put the vectors back in the original order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        batches = [
            self._encode_batch(sorted_texts[i : i + BATCH_SIZE])
            for i in range(0, len(sorted_texts), BATCH_SIZE)
        ]
        sorted_vectors = np.concatenate(batches)

        vectors = np.empty_like(sorted_vectors)
        vectors[order] = sorted_vectors
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single search query. Shape: (dim,)."""
        return self._encode_batch([BGE_QUERY_PREFIX + query])[0]

    def _load(self) -> None:
        # Loaded lazily so creating an EmbeddingModel is instant; the model
        # files (~130 MB) are downloaded from Hugging Face on first use and
        # cached in ~/.cache/huggingface afterwards.
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(
            hf_hub_download(self.model_name, "tokenizer.json")
        )
        self._tokenizer.enable_truncation(MAX_TOKENS)
        self._tokenizer.enable_padding()
        self._session = onnxruntime.InferenceSession(
            hf_hub_download(self.model_name, "onnx/model.onnx")
        )

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        if self._session is None:
            self._load()

        # 1. Tokenize: split text into word pieces and map each to an ID.
        #    Padding makes every text in the batch the same length.
        encodings = self._tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }

        # 2. Run the model. Output shape: (batch, tokens, 384) - one vector
        #    per token. BGE is trained so the first token ([CLS]) summarizes
        #    the whole text, so that vector is the text's embedding.
        token_vectors = self._session.run(None, inputs)[0]
        embeddings = token_vectors[:, 0]

        # 3. Normalize every vector to length 1, so the dot product of two
        #    vectors equals their cosine similarity.
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)
