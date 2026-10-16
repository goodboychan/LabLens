<#
.SYNOPSIS
    Deploys LabLens v1.1 (Robot Episode Investigator) to Google Cloud Run using Google Cloud ADC
    and runs post-deployment verification probes.
#>
param(
    [string]$ProjectId = $env:GOOGLE_CLOUD_PROJECT,
    [string]$ServiceName = "lablens",
    [string]$Region = "us-central1",
    [string]$GeminiLocation = "global",
    [string]$GeminiModelId = "gemini-3-flash-preview"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK ('gcloud') was not found on PATH. Install Google Cloud SDK and run 'gcloud auth login' before executing this script."
    exit 1
}

if (-not $ProjectId) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
}
if (-not $ProjectId -or $ProjectId -eq "(unset)") {
    Write-Error "No GCP Project ID specified. Pass -ProjectId <YOUR_PROJECT_ID> or set GOOGLE_CLOUD_PROJECT."
    exit 1
}

Write-Host "==> Deploying $ServiceName to Cloud Run (Project: $ProjectId, Region: $Region)..." -ForegroundColor Cyan

gcloud services enable `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    artifactregistry.googleapis.com `
    aiplatform.googleapis.com `
    --project $ProjectId

gcloud run deploy $ServiceName `
    --source . `
    --project $ProjectId `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --timeout 120 `
    --max-instances 3 `
    --set-env-vars="LABLENS_USE_ADC=true,GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=$GeminiLocation,GEMINI_MODEL_ID=$GeminiModelId"

$ServiceUrl = (gcloud run services describe $ServiceName --project $ProjectId --region $Region --format="value(status.url)").Trim()
Write-Host "==> Deployed URL: $ServiceUrl" -ForegroundColor Green

Write-Host "==> Running post-deployment verification against $ServiceUrl/api/health ..." -ForegroundColor Cyan
$Health = Invoke-RestMethod -Uri "$ServiceUrl/api/health" -Method Get
Write-Host ($Health | ConvertTo-Json -Depth 4)

if ($Health.status -ne "ok" -or $Health.registered_episodes -lt 16) {
    Write-Error "Post-deployment health check failed!"
    exit 1
}

Write-Host "==> Verifying operator session initialization ($ServiceUrl/api/session)..." -ForegroundColor Cyan
$Session = Invoke-RestMethod -Uri "$ServiceUrl/api/session" -Method Get -SessionVariable WebSession
$Headers = @{
    "X-LabLens-Session" = $Session.session_token
    "X-LabLens-CSRF"    = $Session.csrf_token
}

Write-Host "==> Verifying 4-counter episode catalog ($ServiceUrl/api/catalog)..." -ForegroundColor Cyan
$Catalog = Invoke-RestMethod -Uri "$ServiceUrl/api/catalog" -Method Get -WebSession $WebSession -Headers $Headers
Write-Host "Prepared: $($Catalog.reported_prepared_count) | Registered: $($Catalog.registered_count) | Validated: $($Catalog.validated_count) | Analyzed: $($Catalog.analyzed_count)" -ForegroundColor Green
Write-Host "==> Cloud Run deployment and verification succeeded!" -ForegroundColor Green
