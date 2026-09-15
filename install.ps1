# One-line installer for a dgpt worker node (Windows PowerShell).
#   irm https://<where you host this>/install.ps1 | iex
#   $env:DGPT_SRC="C:\path\dgpt-0.1.0-py3-none-any.whl"; .\install.ps1
$ErrorActionPreference = "Stop"
$src = if ($env:DGPT_SRC) { $env:DGPT_SRC } else { "dgpt @ git+https://github.com/YOUR_ORG/distribute" }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "[dgpt] installing uv"
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
$index = @()
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  $index = @("--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match")
}
Write-Host "[dgpt] installing worker from: $src"
uv tool install --force --python 3.12 @index $src
Write-Host ""
Write-Host "[dgpt] installed. Join a pool with:  dgpt-worker --coordinator http://HOST:8000   (add --token XYZ if required)"
