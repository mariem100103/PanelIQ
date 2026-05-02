# Démarre Postgres (Docker), FastAPI :8000 et Nest :3000 — fenêtres séparées.
# Usage: clic droit → Exécuter avec PowerShell, ou:  powershell -ExecutionPolicy Bypass -File .\start-dev.ps1

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Test-Cmd($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# 1) Postgres via Docker si disponible
$Dabl = Join-Path $Root "DABLyou"
if (Test-Cmd "docker") {
    Write-Host "Docker: démarrage Postgres/Redis..."
    Push-Location $Dabl
    try { docker compose up -d 2>&1 | Out-Host } catch { Write-Warning $_ }
    Pop-Location
    Start-Sleep -Seconds 2
} else {
    Write-Warning "Docker introuvable dans le PATH — assure-toi que PostgreSQL tourne (port 5432)."
}

# 2) FastAPI
$apiDir = Join-Path $Root "api"
$py = @"
Set-Location '$apiDir'
Write-Host 'FastAPI http://127.0.0.1:8000' -ForegroundColor Green
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
"@
Start-Process powershell -WorkingDirectory $apiDir -ArgumentList @("-NoExit", "-Command", $py)

Start-Sleep -Seconds 2

# 3) Nest
$be = Join-Path $Dabl "backend"
$nest = @"
Set-Location '$be'
Write-Host 'NestJS http://localhost:3000' -ForegroundColor Cyan
npm run start:dev
"@
Start-Process powershell -WorkingDirectory $be -ArgumentList @("-NoExit", "-Command", $nest)

Write-Host ""
Write-Host "Deux fenêtres PowerShell ont été ouvertes (FastAPI + Nest)." -ForegroundColor Yellow
Write-Host "Flutter (Chrome): cd DABLyou\mobile ; flutter run -d chrome" -ForegroundColor Gray
