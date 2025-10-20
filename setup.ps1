Param(
  [string]$PythonBin = "python",
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"
if (-Not (Test-Path $VenvDir)) { & $PythonBin -m venv $VenvDir }
& "$VenvDir\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "✅ Ready. Activate with: `"$VenvDir\Scripts\Activate.ps1`""
