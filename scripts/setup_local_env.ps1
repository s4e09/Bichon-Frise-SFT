param(
    [string]$CondaEnv = "bixiong-lora"
)

$ErrorActionPreference = "Stop"

$conda = "E:\Anaconda\condabin\conda.bat"
if (-not (Test-Path $conda)) {
    throw "Conda not found at $conda"
}

& $conda env list | Select-String "^\s*$CondaEnv\s" | Out-Null
if ($LASTEXITCODE -ne 0) {
    & $conda create -n $CondaEnv python=3.11 pip -y
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$env:PYTHONNOUSERSITE = "1"
& $conda run -n $CondaEnv python -m pip install --ignore-installed `
    --extra-index-url https://download.pytorch.org/whl/cu118 `
    torch==2.7.1+cu118 `
    torchvision==0.22.1+cu118 `
    torchaudio==2.7.1+cu118 `
    transformers `
    peft `
    accelerate `
    datasets `
    safetensors `
    sentencepiece

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
