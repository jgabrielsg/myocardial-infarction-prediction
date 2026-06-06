# Install PyTorch with CUDA 12.4 for NVIDIA GPUs (RTX 3050+)
# Driver CUDA 13.x is backward-compatible with cu124 wheels.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPip) {
    $pip = $VenvPip
    $python = $VenvPython
    Write-Host "Using project venv: $ProjectRoot\.venv"
} else {
    $pip = "pip"
    $python = "python"
}

Write-Host "Installing PyTorch with CUDA 12.4..."
& $pip uninstall -y torch torchvision torchaudio 2>$null
& $pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "Verifying CUDA..."
& $python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host ""
Write-Host "Installing Optuna..."
& $pip install "optuna>=3.6.0"

Write-Host "Done."
