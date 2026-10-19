#!/bin/sh
# Qwiklabs Hotlab Pre-Warming Startup Script (go/qwiklabs-pre-warming)
# Warm window: 5 hours | Recommended pre-warm lead time: 30 minutes
# Any non-zero exit value causes the Qwiklabs startup script to fail.
set -eu

# Environment Variables
PROJECT_ID=$1
REGION=$2
ZONE=$3
ACCOUNT="${PROJECT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

# Set the Platform Project, Region, and Zone
gcloud config set project "$PROJECT_ID"
gcloud config set compute/region "$REGION"
gcloud config set compute/zone "$ZONE"
echo "PROJECT #1 ${PROJECT_ID} (${REGION} / ${ZONE})"

# Pre-enable Google Cloud APIs required by Cluster Toolkit and AlphaEvolve HPC
gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  batch.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  notebooks.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project="${PROJECT_ID}" \
  --account="$ACCOUNT"

# Verify regional CPU quota availability
gcloud compute regions describe "${REGION}" \
  --project="${PROJECT_ID}" \
  --account="$ACCOUNT" \
  --flatten="quotas" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep -E "METRIC|^CPUS |^INSTANCES |^N2_CPUS " || true
