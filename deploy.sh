#!/bin/bash

# Aurora QA System - Google Cloud Run Deployment Script

set -e

PROJECT_ID="aurora-478208"
SERVICE_NAME="aurora-qa-system"
REGION="us-central1"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:latest .

echo "Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}:latest

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME}:latest \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --set-env-vars "OPENAI_API_KEY=${OPENAI_API_KEY},AURORA_API_URL=${AURORA_API_URL}" \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0

echo "Deployment complete!"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format='value(status.url)'

