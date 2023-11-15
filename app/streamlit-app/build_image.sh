#!/usr/bin/env bash

GCLOUD_PROJECT_ID=$(gcloud config get-value project)
GCLOUD_REGION="us-central1"
GCLOUD_ARTIFACTS_REPOSITORY="prefect"
IMAGE_NAME="streamlit-app"
gcloud builds submit --async \
  --region=${GCLOUD_REGION} \
  --tag "${GCLOUD_REGION}-docker.pkg.dev/${GCLOUD_PROJECT_ID}/${GCLOUD_ARTIFACTS_REPOSITORY}/${IMAGE_NAME}" .
