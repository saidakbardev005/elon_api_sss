FROM python:3.10-slim

# App papkasi
WORKDIR /app

# Build‑time kerakli paketlar
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Reqs va gunicorn
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install gunicorn

# Qolgan kod
COPY . .

# Port deklaratsiya
EXPOSE 5000

# Default CMD — gunicorn bilan run
CMD ["python", "app.py"]
