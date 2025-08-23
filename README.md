# SLA Sentence Retrieval Evaluation on Checkpoints

## Models
- https://huggingface.co/tktung/sla_cpt

## Dataset
- https://huggingface.co/datasets/openlanguagedata/flores_plus

## Sentence retrieval task reference

- https://aclanthology.org/2024.loresmt-1.20.pdf

## Folder structure

`job_scripts`: Slurm job scripts


## Usage

### Using python

```
usage: sr_experiment.py [-h] --model_path MODEL_PATH --base BASE --target TARGET [--device DEVICE]
                        [--artifact-out ARTIFACT_OUT] [--margin-variant MARGIN_VARIANT]

Sentence retrieval task with plotting

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Folder with checkpoints or Hugging Face model ID
  --base BASE           Base language ISO code
  --target TARGET       Target language ISO code
  --device DEVICE       Device to use (cuda or cpu)
  --artifact-out ARTIFACT_OUT
                        Folder to save JSON artifacts and plot
  --margin-variant MARGIN_VARIANT
                        Margin variant for scoring
```


### Using slurm

Run an experiment

```
cd job_scripts
sbatch sre.sh <path_to_env_file>
```

Example
```
sbatch sre.sh envs/uccix_llama2-13B_eng-gle.sh
```

Run all experiments in `envs` folder

```
cd job_scripts
sbatch sre_all.sh
```

