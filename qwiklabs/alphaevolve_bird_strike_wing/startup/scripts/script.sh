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

# Set the Platform Project
gcloud config set project "$PROJECT_ID"
echo "PROJECT #1 ${PROJECT_ID} (${REGION} / ${ZONE})"

gcloud auth list
export PROJECT_ID=$(gcloud config get-value project)
export REGION="us-central1"
export ZONE="us-central1-a"
export BUCKET_NAME="${PROJECT_ID}-alphaevolve-bucket"

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"

echo "Project ID: ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Bucket:     gs://${BUCKET_NAME}"
gcloud storage buckets create "gs://${BUCKET_NAME}" \
  --project="${PROJECT_ID}" \
  --location="${REGION}" \
  --uniform-bucket-level-access
gcloud services enable compute.googleapis.com --project="${PROJECT_ID}" --account="$ACCOUNT"
gcloud compute regions describe "${REGION}" \
  --project="${PROJECT_ID}" \
  --flatten="quotas" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" \
  | grep -E "METRIC|^CPUS |^INSTANCES |^N2_CPUS "
cd "${HOME}"
git clone https://github.com/WyattGorman/scientific-computing-examples.git
cd "${HOME}/scientific-computing-examples/alphaevolve-hpc"
ls -la user_examples/bird_strike_wing/
