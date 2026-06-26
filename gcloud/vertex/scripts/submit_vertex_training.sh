#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-dtumlops-499809}"
REGION="${REGION:-europe-west1}"
IMAGE_URI="${IMAGE_URI:-europe-west1-docker.pkg.dev/${PROJECT_ID}/dtumlops/blackjack-train:latest}"
JOB_SPEC="${JOB_SPEC:-gcloud/vertex/blackjack_train_custom_job.yaml}"
BUILD_CONFIG="${BUILD_CONFIG:-gcloud/vertex/cloudbuild.yaml}"
JOB_DISPLAY_NAME="${JOB_DISPLAY_NAME:-blackjack-train}"

gcloud builds submit \
  --project "${PROJECT_ID}" \
  --config "${BUILD_CONFIG}" \
  --substitutions "_TRAIN_IMAGE=${IMAGE_URI}"

gcloud ai custom-jobs create \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --display-name "${JOB_DISPLAY_NAME}" \
  --config "${JOB_SPEC}"
