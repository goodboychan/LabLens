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
    [string]$GeminiModelId = "gemini-4-argon"
)

$ErrorActionPreference = "Continue"

$LocalGcloudBin = "C:\Users\admin\google-cloud-sdk-install\google-cloud-sdk\bin"
if ((Test-Path $LocalGcloudBin) -and ($env:PATH -notlike "*$LocalGcloudBin*")) {
    $env:PATH = "$LocalGcloudBin;$env:PATH"
}
if (-not $env:CLOUDSDK_PYTHON -and (Test-Path "C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe")) {
    $env:CLOUDSDK_PYTHON = "C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe"
}

if (-not (Get-Command gcloud.cmd -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK ('gcloud.cmd') was not found on PATH. Install Google Cloud SDK and run 'gcloud auth login' before executing this script."
    exit 1
}

if (-not $ProjectId -or $ProjectId -eq "(unset)" -or $ProjectId -eq "kr-ai-hackathon26gmp-2225") {
    $AvailProjects = @(cmd /c "gcloud.cmd projects list --format=value(projectId) 2>nul")
    if ($AvailProjects -contains "h2gdm-hack26icn-2644") {
        $ProjectId = "h2gdm-hack26icn-2644"
    } elseif ($AvailProjects.Count -gt 0 -and $AvailProjects[0]) {
        $ProjectId = $AvailProjects[0].Trim()
    } else {
        $ProjectId = (cmd /c "gcloud.cmd config get-value project 2>nul").Trim()
    }
}
if (-not $ProjectId -or $ProjectId -eq "(unset)") {
    Write-Error "No GCP Project ID specified. Pass -ProjectId <YOUR_PROJECT_ID> or set GOOGLE_CLOUD_PROJECT."
    exit 1
}
cmd /c "gcloud.cmd config set project $ProjectId --quiet >nul 2>&1"
cmd /c "gcloud.cmd auth application-default set-quota-project $ProjectId --quiet >nul 2>&1"

Write-Host "==> Deploying $ServiceName to Cloud Run (Project: $ProjectId, Region: $Region)..." -ForegroundColor Cyan

cmd /c "gcloud.cmd services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com aiplatform.googleapis.com --project $ProjectId --quiet"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to enable required GCP services."
    exit $LASTEXITCODE
}

$OperatorSecret = $env:LABLENS_OPERATOR_SECRET
if (-not $OperatorSecret) {
    $RngBytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($RngBytes)
    $OperatorSecret = [Convert]::ToBase64String($RngBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

cmd /c "gcloud.cmd run deploy $ServiceName --source . --project $ProjectId --region $Region --platform managed --allow-unauthenticated --memory 2Gi --cpu 2 --timeout 120 --max-instances 3 --quiet --set-env-vars=""LABLENS_USE_ADC=true,GOOGLE_CLOUD_PROJECT=$ProjectId,GOOGLE_CLOUD_LOCATION=$GeminiLocation,GEMINI_MODEL_ID=$GeminiModelId,LABLENS_OPERATOR_SECRET=$OperatorSecret,LABLENS_SESSION_TOKEN=$OperatorSecret"""
if ($LASTEXITCODE -ne 0) {
    Write-Error "Cloud Run deployment failed."
    exit $LASTEXITCODE
}

$ServiceUrl = (cmd /c "gcloud.cmd run services describe $ServiceName --project $ProjectId --region $Region --format=value(status.url) 2>nul").Trim()
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
