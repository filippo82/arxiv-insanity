#!/usr/bin/env bash

GCLOUD_REGION="us-central1"
SERVICE_NAME="streamlit-app"
gcloud run deploy $SERVICE_NAME \
  --image=us-central1-docker.pkg.dev/arxiv-insanity-v1-404114/prefect/streamlit-app:latest \
  --region=${GCLOUD_REGION} \
  --set-env-vars=VESPA_ENDPOINT=http://34.41.66.135:80 \
  --allow-unauthenticated \
  --memory=4Gi
