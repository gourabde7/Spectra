FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY . .
ENV NUMBA_CACHE_DIR=/tmp/numba HOME=/tmp PYTHONUNBUFFERED=1
RUN python -c "import fec; fec.warm_up()" && chmod -R a+rwX /tmp/numba
EXPOSE 7860
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-7860} --server.address=0.0.0.0"]
