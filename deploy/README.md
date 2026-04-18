# deploy

Cloud deployment utilities for LoyaltyLens — AWS SageMaker and GCP Vertex AI.
All code is additive; no existing module, Dockerfile, or `docker-compose.yml` is touched.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | `DeploySettings` — single source of truth for all cloud config |
| `cloud_storage.py` | Route artifact uploads to S3 or GCS based on `DEPLOY_TARGET` |
| `sagemaker_deploy.py` | Deploy / invoke / teardown propensity model on SageMaker |
| `vertex_deploy.py` | Deploy propensity to Vertex AI Prediction; index offers in Vector Search |
| `.env.sagemaker` | Template env file for SageMaker deployments |
| `.env.vertex` | Template env file for Vertex AI deployments |

## Switching targets

One env var controls everything:

```
DEPLOY_TARGET=local       # default — no cloud calls
DEPLOY_TARGET=sagemaker   # AWS SageMaker
DEPLOY_TARGET=vertex      # GCP Vertex AI
```

---

## AWS SageMaker — full setup

### 1. Install the AWS CLI

Download from [https://aws.amazon.com/cli/](https://aws.amazon.com/cli/) and verify:

```powershell
aws --version
```

### 2. Create an IAM user with programmatic access

1. Open **IAM → Users → Create user** in the AWS console.
2. Attach these managed policies directly:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
3. Go to **Security credentials → Create access key** → choose **CLI**.
4. Copy `Access key ID` and `Secret access key`.

### 3. Configure credentials locally

```powershell
aws configure
# AWS Access Key ID:     AKIA...
# AWS Secret Access Key: ...
# Default region:        us-east-1
# Default output format: json
```

This writes to `~/.aws/credentials` and `~/.aws/config`. Alternatively, add directly to the root `.env`:

```dotenv
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

### 4. Create a SageMaker execution role

SageMaker needs its own IAM role (separate from your user) to launch training jobs and endpoints.

1. **IAM → Roles → Create role**
2. Trusted entity: **AWS service → SageMaker**
3. Attach policy: `AmazonSageMakerFullAccess`
4. Name it `SageMakerRole` (or anything — copy the ARN)

### 5. Create an S3 bucket

```powershell
aws s3 mb s3://loyaltylens-artifacts --region us-east-1
```

Bucket name must be globally unique. Update `S3_BUCKET` in `deploy/.env.sagemaker`.

### 6. Install SDK deps

```powershell
uv pip install boto3 sagemaker skl2onnx --python .venv\Scripts\python.exe
```

### 7. Verify access

```powershell
python -c "import boto3; print(boto3.client('sts').get_caller_identity())"
```

Should print your `UserId`, `Account`, and `Arn`.

### 8. Configure `deploy/.env.sagemaker`

```dotenv
DEPLOY_TARGET=sagemaker
AWS_REGION=us-east-1
SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerRole
SAGEMAKER_INSTANCE_TYPE=ml.t2.medium
SAGEMAKER_ENDPOINT_NAME=loyaltylens-propensity
S3_BUCKET=loyaltylens-artifacts
SAVE_OUTPUTS_TO_CLOUD=true
```

### 9. Deploy

```powershell
python deploy/sagemaker_deploy.py --action deploy --model-path models/propensity_v1.pt
```

This will:

1. Create the S3 bucket `loyaltylens-artifacts` if it does not exist
2. Write `propensity_model/inference.py` (PyTorch serving entry point)
3. Package `propensity_v1.pt` as `model.tar.gz` and upload to S3
4. Deploy a SageMaker `PyTorchModel` endpoint (~10 minutes)

### 10. Invoke

```powershell
python deploy/sagemaker_deploy.py --action invoke `
  --config deploy/.env.sagemaker `
  --payload '{\"recency_days\": 3, \"frequency_30d\": 5, \"monetary_90d\": 120, \"offer_redemption_rate\": 0.4, \"engagement_score\": 0.7}'
```

Returns:
```json
{"propensity_score": 0.82, "model_version": "1"}
```

### 11. Teardown

```powershell
python deploy/sagemaker_deploy.py --action teardown --config deploy/.env.sagemaker
```

Deletes the endpoint, endpoint config, and model. **Do this when done — SageMaker charges for endpoint uptime.**

---

## GCP Vertex AI — full setup

### 1. Install the Google Cloud CLI (gcloud)

Download from [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install) and verify:

```powershell
gcloud --version
```

### 2. Create or select a GCP project

```powershell
gcloud projects create loyaltylens-demo --name="LoyaltyLens Demo"
gcloud config set project loyaltylens-demo
```

Or use an existing project:

```powershell
gcloud config set project YOUR_EXISTING_PROJECT_ID
```

### 3. Enable required APIs

```powershell
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

### 4. Create a service account

```powershell
gcloud iam service-accounts create loyaltylens-sa `
  --display-name="LoyaltyLens Deploy SA"
```

Grant it the required roles:

```powershell
$PROJECT = gcloud config get-value project

gcloud projects add-iam-policy-binding $PROJECT `
  --member="serviceAccount:loyaltylens-sa@$PROJECT.iam.gserviceaccount.com" `
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT `
  --member="serviceAccount:loyaltylens-sa@$PROJECT.iam.gserviceaccount.com" `
  --role="roles/storage.objectAdmin"
```

### 5. Authenticate locally

**Option A — Application Default Credentials (recommended for local dev):**

```powershell
gcloud auth application-default login
```

Opens a browser; credentials are written to `~/.config/gcloud/application_default_credentials.json`.

**Option B — Service account key (CI/CD or machines without a browser):**

```powershell
gcloud iam service-accounts keys create deploy/sa-key.json `
  --iam-account="loyaltylens-sa@$PROJECT.iam.gserviceaccount.com"
```

Then set:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS=deploy/sa-key.json
```

> **Never commit `sa-key.json`** — add it to `.gitignore`.

### 6. Create a GCS bucket

```powershell
gsutil mb -l us-central1 gs://loyaltylens-artifacts
```

### 7. Install SDK deps

```powershell
uv pip install google-cloud-aiplatform google-cloud-storage --python .venv\Scripts\python.exe
```

### 8. Verify access

```powershell
python -c "
from google.cloud import storage
client = storage.Client()
print('Project:', client.project)
"
```

### 9. Configure `deploy/.env.vertex`

```dotenv
DEPLOY_TARGET=vertex
GCP_PROJECT=loyaltylens-demo
GCP_REGION=us-central1
GCS_BUCKET=loyaltylens-artifacts
VERTEX_ENDPOINT_ID=      # filled in after deploy-model
VERTEX_INDEX_ID=         # filled in after deploy-index
SAVE_OUTPUTS_TO_CLOUD=true
```

### 10. Deploy model (Module 2 — propensity)

Upload the model artifact first:

```powershell
gsutil cp models/propensity_v1.pt gs://loyaltylens-artifacts/models/propensity_v1.pt
```

Then deploy:

```powershell
python deploy/vertex_deploy.py --action deploy-model `
  --config deploy/.env.vertex `
  --gcs-uri gs://loyaltylens-artifacts/models/propensity_v1.pt
```

Copy the printed endpoint resource name into `VERTEX_ENDPOINT_ID`.

### 11. Predict

```powershell
python deploy/vertex_deploy.py --action predict `
  --config deploy/.env.vertex `
  --payload '{\"recency_days\": 3, \"frequency_30d\": 5, \"monetary_90d\": 120, \"offer_redemption_rate\": 0.4, \"engagement_score\": 0.7}'
```

### 12. Deploy Vector Search index (Module 3 stretch goal)

Build an embeddings JSONL file (one object per line):

```json
{"id": "O001", "embedding": [0.12, -0.34, 0.56, ...]}
```

Upload and deploy:

```powershell
gsutil cp data/offer_embeddings.jsonl gs://loyaltylens-artifacts/embeddings/offers.jsonl

python deploy/vertex_deploy.py --action deploy-index `
  --config deploy/.env.vertex `
  --gcs-uri gs://loyaltylens-artifacts/embeddings/offers.jsonl
```

Copy the printed endpoint resource name into `VERTEX_INDEX_ID`, then query:

```powershell
python deploy/vertex_deploy.py --action query `
  --config deploy/.env.vertex `
  --embedding '[0.12, -0.34, 0.56, ...]'
```

---

## Cost estimate

| Target | Instance | On-demand hourly |
| --- | --- | --- |
| SageMaker | `ml.t2.medium` | ~$0.056 / hr |
| Vertex AI Prediction | `n1-standard-2` | ~$0.095 / hr |
| Vertex Vector Search | Managed index | ~$0.10 / hr (small index) |

**Teardown endpoints when not in use.** Both platforms charge for endpoint uptime regardless of traffic.

---

## End-to-end pipeline (local)

```powershell
python shared/pipeline.py --customer-id C001
```

Uses all 6 modules without any cloud target. Cloud deployers are separate and do not need to be running for the pipeline to work.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: boto3` | SDK not installed | `uv pip install boto3 sagemaker --python .venv\Scripts\python.exe` |
| `ModuleNotFoundError: google.cloud.aiplatform` | SDK not installed | `uv pip install google-cloud-aiplatform google-cloud-storage --python .venv\Scripts\python.exe` |
| `ValidationError: DEPLOY_TARGET` | Invalid value | Must be `local`, `sagemaker`, or `vertex` |
| `botocore.exceptions.NoCredentialsError` | No AWS creds | Run `aws configure` or set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in `.env` |
| `sagemaker.exceptions.UnexpectedStatusException` | Endpoint failed to start | Check CloudWatch Logs → `/aws/sagemaker/Endpoints/loyaltylens-propensity` |
| `google.auth.exceptions.DefaultCredentialsError` | No GCP creds | Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS` |
| `google.api_core.exceptions.PermissionDenied` | Missing IAM role | Add `roles/aiplatform.user` to the service account |
| `AccessDeniedException` on S3 | Bucket policy | Check `SAGEMAKER_ROLE_ARN` has `AmazonS3FullAccess` on `S3_BUCKET` |
