FROM python:3.11-slim

WORKDIR /app

# Copier requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]