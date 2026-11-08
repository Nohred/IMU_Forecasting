FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/src \
    MPLBACKEND=Agg

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY checkpoints/ ./checkpoints/
COPY model_repository/ ./model_repository/

CMD ["python", "-m", "pytest", "tests"]
