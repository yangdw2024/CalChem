$ErrorActionPreference = "Stop"
. "$PSScriptRoot\..\2026-11-10_round2\probes\env.ps1"

Write-Host "=== Running Full Pytest tests/unit ==="
& $env:PYO3_PYTHON -B -c "import os,tempfile,pytest; tempfile.tempdir=os.environ['TEMP']; raise SystemExit(pytest.main(['tests/unit','-q','-p','no:cacheprovider']))"
Write-Host "Full Pytest exit code: $LASTEXITCODE"
