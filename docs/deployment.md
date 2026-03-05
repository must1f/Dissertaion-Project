# Deployment Guide

## Production Architecture

```
                    ┌─────────────┐
                    │   Nginx     │
                    │  (Reverse   │
                    │   Proxy)    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Frontend   │ │   Backend   │ │  Backend    │
    │   (Static)  │ │  (Uvicorn)  │ │  (Worker)   │
    └─────────────┘ └──────┬──────┘ └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  PostgreSQL │
                    │ +TimescaleDB│
                    └─────────────┘
```

## Docker Deployment

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/pinn
      - MODELS_DIR=/app/Models
      - RESULTS_DIR=/app/results
    depends_on:
      - db
    volumes:
      - ./Models:/app/Models:ro
      - ./results:/app/results:ro

  db:
    image: timescale/timescaledb:latest-pg14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=pinn
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

### Backend Dockerfile

**backend/Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML core
COPY src/ /app/src/
RUN pip install -e /app/

# Copy backend
COPY backend/ /app/backend/

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

**frontend/Dockerfile:**
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

### Nginx Configuration

**frontend/nginx.conf:**
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Frontend routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API proxy
    location /api {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket proxy
    location /ws {
        proxy_pass http://backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
}
```

## Cloud Deployment

### AWS Deployment

1. **Frontend**: Deploy to S3 + CloudFront
2. **Backend**: Deploy to ECS or Lambda
3. **Database**: Use RDS with TimescaleDB

```bash
# Build and push backend image
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URL
docker build -t pinn-backend -f backend/Dockerfile .
docker tag pinn-backend:latest $ECR_URL/pinn-backend:latest
docker push $ECR_URL/pinn-backend:latest

# Deploy frontend to S3
cd frontend
npm run build
aws s3 sync dist/ s3://your-bucket-name/ --delete
aws cloudfront create-invalidation --distribution-id $CF_ID --paths "/*"
```

### GCP Deployment

1. **Frontend**: Deploy to Cloud Storage + Cloud CDN
2. **Backend**: Deploy to Cloud Run
3. **Database**: Use Cloud SQL

```bash
# Deploy backend to Cloud Run
gcloud run deploy pinn-backend \
  --image gcr.io/$PROJECT_ID/pinn-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Environment Variables

### Production Backend

```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/pinn

# Paths
MODELS_DIR=/app/Models
RESULTS_DIR=/app/results
SRC_DIR=/app/src

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# CORS
CORS_ORIGINS=https://your-domain.com

# Security (optional)
SECRET_KEY=your-secret-key
API_KEY=your-api-key
```

### Production Frontend

```env
VITE_API_URL=https://api.your-domain.com
VITE_WS_URL=wss://api.your-domain.com
```

## Performance Optimization

### Backend

1. **Gunicorn with Uvicorn workers**
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Model caching**
- Models are cached in ModelRegistry singleton
- Use Redis for distributed caching if needed

3. **Database connection pooling**
- SQLAlchemy pool_size configured in database.py

### Frontend

1. **Code splitting**
```typescript
// Lazy load pages
const Dashboard = lazy(() => import('./pages/Dashboard'))
```

2. **Asset optimization**
- Vite handles tree-shaking and minification
- Consider CDN for large dependencies

## Monitoring

### Health Checks

**Backend health endpoint:**
```python
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow()}
```

### Logging

Configure structured logging:
```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        })
```

### Metrics

Consider adding Prometheus metrics:
```python
from prometheus_client import Counter, Histogram

predictions_total = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
```

## Security Checklist

- [ ] Enable HTTPS
- [ ] Set secure CORS origins
- [ ] Use environment variables for secrets
- [ ] Enable rate limiting
- [ ] Validate all inputs
- [ ] Keep dependencies updated
- [ ] Enable security headers (CSP, HSTS)
