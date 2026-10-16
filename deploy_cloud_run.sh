#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${1:-${GOOGLE_CLOUD_PROJECT:-}}"
SERVICE_NAME="${SERVICE_NAME:-lablens}"
REGION="${REGION:-us-central1}"
GEMINI_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
GEMINI_MODEL_ID="${GEMINI_MODEL_ID:-gemini-4-argon}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud CLI is not installed or not on PATH." >&2
  exit 1
fi

if [[ -z "${PROJECT_ID}" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [[ -z "${PROJECT_ID}" || "${PROJECT_ID}" == "(unset)" ]]; then
  echo "Usage: ./deploy_cloud_run.sh <GCP_PROJECT_ID>" >&2
  exit 1
fi

echo "==> Enabling required Google Cloud APIs for project ${PROJECT_ID}..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  --project "${PROJECT_ID}"

OPERATOR_SECRET="${LABLENS_OPERATOR_SECRET:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')}"

echo "==> Deploying ${SERVICE_NAME} to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --max-instances 3 \
  --set-env-vars="LABLENS_USE_ADC=true,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${GEMINI_LOCATION},GEMINI_MODEL_ID=${GEMINI_MODEL_ID},LABLENS_OPERATOR_SECRET=${OPERATOR_SECRET},LABLENS_SESSION_TOKEN=${OPERATOR_SECRET}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')"
echo "==> Deployed URL: ${SERVICE_URL}"

echo "==> Verifying ${SERVICE_URL}/api/health ..."
curl -fsSL "${SERVICE_URL}/api/health" | python3 -m json.tool
echo "==> Cloud Run deployment and verification complete!"
