param(
    [string]$CondaEnv = "bixiong-lora",
    [string]$Config = "configs/local_peft_2024plus.json",
    [switch]$DryRun,
    [int]$MaxTrainSamples = 0,
    [int]$MaxValSamples = 0
)

$ErrorActionPreference = "Stop"

$conda = "E:\Anaconda\condabin\conda.bat"
if (-not (Test-Path $conda)) {
    throw "Conda not found at $conda"
}

$scriptArgs = @("scripts/train_local_peft.py", "--config", $Config)
if ($DryRun) {
    $scriptArgs += "--dry-run"
}
if ($MaxTrainSamples -gt 0) {
    $scriptArgs += @("--max-train-samples", [string]$MaxTrainSamples)
}
if ($MaxValSamples -gt 0) {
    $scriptArgs += @("--max-val-samples", [string]$MaxValSamples)
}

$env:PYTHONNOUSERSITE = "1"
& $conda run -n $CondaEnv python @scriptArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
