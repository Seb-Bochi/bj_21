#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-dtumlops-499809}"
REGION="${REGION:-europe-west1}"
SERVICE="${SERVICE:-blackjack-predictor-api}"

SERVICE_URL="$(
  gcloud run services describe "${SERVICE}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format "value(status.url)"
)"

if [[ -z "${SERVICE_URL}" ]]; then
  echo "Cloud Run service ${SERVICE} has no URL."
  exit 1
fi

if ID_TOKEN="$(gcloud auth print-identity-token 2>/dev/null)"; then
  curl --fail --silent --show-error -H "Authorization: Bearer ${ID_TOKEN}" "${SERVICE_URL}/health"
  echo
  echo "Verified Cloud Run health endpoint: ${SERVICE_URL}"
else
  SERVICE_STATUS="$(
    gcloud run services describe "${SERVICE}" \
      --project "${PROJECT_ID}" \
      --region "${REGION}" \
      --format "value(status.conditions[0].status)"
  )"
  test "${SERVICE_STATUS}" = "True"
  echo "Verified Cloud Run service is Ready: ${SERVICE_URL}"
fi
