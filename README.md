# SLA Sentence Retrieval Evaluation on Checkpoints

## Models
- https://huggingface.co/tktung/sla_cpt

## Dataset
- https://huggingface.co/datasets/openlanguagedata/flores_plus

## Sentence retrieval task reference

- https://aclanthology.org/2024.loresmt-1.20.pdf

## Folder structure

`job_scripts`: Slurm job scripts, logs folder for scripts should be in here as well
`json_artifacts`: Json artifacts generated from running `sr_experiment.py`
`plot_artifacts`: Plot artifacts generated from `sr_plot.py`

## Usage

Note: To access gated hugging face repo run `hf auth login` on the command line first

### Generate JSON artifacts

```
Usage:
python sr_experiment.py [-h] --model_path MODEL_PATH --base BASE --target TARGET [--device DEVICE]
                             [--json-artifact-out JSON_ARTIFACT_OUT] [--plot-artifact-out PLOT_ARTIFACT_OUT]
```

### Generate Plot artifacts
```
Usage: 
python sr_plot.py [-h] --json JSON [JSON ...] [--plot-artifact-out PLOT_ARTIFACT_OUT]
```