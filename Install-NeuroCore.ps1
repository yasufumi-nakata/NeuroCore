$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& py -3 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.10+ with py launcher is required" }
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip failed" }
& $python -m pip install ".[desktop]"
if ($LASTEXITCODE -ne 0) { throw "NeuroCore install failed" }
Write-Host "Installation completed."
