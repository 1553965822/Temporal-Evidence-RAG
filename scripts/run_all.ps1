$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Conda = $env:CONDA_EXE
if (-not $Conda) {
  $Conda = "conda"
}

& $Conda run -n paper_rag python -m compileall src scripts
& $Conda run -n paper_rag python scripts\run_paper_experiment.py --mode measured
& $Conda run -n paper_rag python scripts\run_component_experiments.py --mode all --output-suffix run_all_real --evidence-dataset GrassRisk
