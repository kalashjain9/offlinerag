# ============================================
# OfflineRAG - PowerShell Launch Script
# ============================================
# This script starts both the backend and frontend servers

param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly,
    [switch]$Install
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  OfflineRAG - Production RAG Chatbot" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Check-Prerequisites {
    Write-Host "[*] Checking prerequisites..." -ForegroundColor Yellow
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "  [OK] Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "  [X] Python not found. Please install Python 3.10+" -ForegroundColor Red
        exit 1
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "  [OK] Node.js: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "  [X] Node.js not found. Please install Node.js 18+" -ForegroundColor Red
        exit 1
    }
    
    # Check Ollama
    try {
        $ollamaRunning = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction SilentlyContinue
        Write-Host "  [OK] Ollama is running" -ForegroundColor Green
    } catch {
        Write-Host "  [!] Ollama not running. Please start Ollama first: ollama serve" -ForegroundColor Yellow
    }
    
    Write-Host ""
}

function Install-Dependencies {
    Write-Host "[*] Installing dependencies..." -ForegroundColor Yellow
    
    # Backend
    Write-Host "  Installing backend dependencies..." -ForegroundColor Gray
    Push-Location backend
    
    if (-not (Test-Path "venv")) {
        python -m venv venv
    }
    
    & ".\venv\Scripts\Activate.ps1"
    pip install -r requirements.txt -q
    
    Pop-Location
    
    # Frontend
    Write-Host "  Installing frontend dependencies..." -ForegroundColor Gray
    Push-Location frontend
    npm install --silent
    Pop-Location
    
    Write-Host "  [OK] Dependencies installed" -ForegroundColor Green
    Write-Host ""
}

function Start-Backend {
    Write-Host "[*] Starting backend server..." -ForegroundColor Yellow
    
    Push-Location backend
    
    # Activate virtual environment
    & ".\venv\Scripts\Activate.ps1"
    
    # Set environment variables
    $env:PYTHONPATH = Get-Location
    
    # Start uvicorn
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"
    
    Pop-Location
    
    Write-Host "  [OK] Backend running at http://localhost:8000" -ForegroundColor Green
}

function Start-Frontend {
    Write-Host "[*] Starting frontend server..." -ForegroundColor Yellow
    
    Push-Location frontend
    
    Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run", "dev"
    
    Pop-Location
    
    Write-Host "  [OK] Frontend running at http://localhost:3000" -ForegroundColor Green
}

# Main execution
Write-Header

if ($Install) {
    Check-Prerequisites
    Install-Dependencies
    Write-Host "[OK] Installation complete!" -ForegroundColor Green
    Write-Host "Run '.\start.ps1' to start the application" -ForegroundColor Gray
    exit 0
}

Check-Prerequisites

if (-not $FrontendOnly) {
    Start-Backend
    Start-Sleep -Seconds 2
}

if (-not $BackendOnly) {
    Start-Frontend
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  OfflineRAG is now running!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all servers" -ForegroundColor Gray
Write-Host ""

# Keep script running
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host "Shutting down..." -ForegroundColor Yellow
}
