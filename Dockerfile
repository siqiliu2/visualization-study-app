FROM python:3.12-slim

WORKDIR /app

# Install the Graphviz system executable used by Digraph.pipe()
RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000

CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --timeout 120 app:app

RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz