# Stable Overall Rerun

Use the real MiniCPM/RAG evaluation entry point with a fresh output suffix.

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

python scripts/run_minicpm_rag_evaluation.py `
  --dataset GrassRisk `
  --train-dataset GrassRisk `
  --tune-dataset GrassRisk `
  --split test `
  --decision-mode hybrid `
  --label-style numeric `
  --calibration-objective f1 `
  --output-suffix "real_grassrisk_$stamp"

python scripts/run_minicpm_rag_evaluation.py `
  --dataset CUADRisk `
  --train-dataset CUADRisk `
  --tune-dataset CUADRisk `
  --split test `
  --decision-mode hybrid `
  --label-style numeric `
  --calibration-objective f1 `
  --output-suffix "real_cuadrisk_$stamp"
```

Read only the newly created directory under `outputs/minicpm_rag_eval/`.
