FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only essential files
COPY model.py .
COPY train.py .
COPY app.py .

# Copy folders
COPY data/ data/
COPY models/ models/
COPY templates/ templates/

# No need to recreate folders you already copied
EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
