FROM python:3.11-slim

WORKDIR /app

# كسر الـ cache
ARG CACHEBUST=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]