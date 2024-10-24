FROM python:3.10-slim
ARG build_type
WORKDIR /app
ADD embed-model.tar.gz $build_type/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY $build_type/ .env ./
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
EXPOSE 8080
RUN ls
CMD ["python", "app.py"]