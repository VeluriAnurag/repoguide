# Server recipe for the public website on Hugging Face Spaces.
# Runs Ollama (Llama 3.2) and the Streamlit app in one container.
FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

# Hugging Face Spaces runs containers as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_KEEP_ALIVE=-1 \
    REPOGUIDE_PUBLIC=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download Llama 3.2 while building, so visitors don't wait for it.
RUN (ollama serve > /dev/null 2>&1 &) && sleep 5 && ollama pull llama3.2

COPY --chown=user . .

# Download the BGE embedding model while building too.
RUN python -c "from app.embeddings import EmbeddingModel; EmbeddingModel().encode_query('warm up')"

EXPOSE 7860
CMD ["bash", "start.sh"]
