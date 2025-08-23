"""
Sentence retrieval task accuracy on FLORES-PRO dataset

Ref 1: https://aclanthology.org/2024.loresmt-1.20.pdf
Ref 2: https://arxiv.org/pdf/1811.01136
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor

from datasets import load_dataset, concatenate_datasets
from datasets.arrow_dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map="auto") -> AutoModelForCausalLM:
    """Load a model"""
    print(f"Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def load_flores(dataset: str) -> list[str]:
    FLORES = "openlanguagedata/flores_plus"
    ds = load_dataset(FLORES, dataset)
    ds = ds["dev"]
    return [d["text"] for d in ds]


def cosine_similarity(first: Tensor, second: Tensor) -> Tensor:
    first_normalized: Tensor = first / torch.linalg.norm(first, ord=2, dim=1, keepdim=True)
    second_normalized: Tensor = second / torch.linalg.norm(second, ord=2, dim=1, keepdim=True)
    return first_normalized @ second_normalized.T


def margin_based_scoring(
    first: Tensor, second: Tensor, k: int = 4, variant: str = "ratio", eps: float = 1e-6
) -> Tensor:
    a = cosine_similarity(first, second)

    if variant == "absolute":
        return a

    nn_k_row = torch.topk(input=a, k=min(k, a.size(1)), dim=1).values
    nn_k_col = torch.topk(input=a, k=min(k, a.size(0)), dim=0).values

    row_mean = nn_k_row.mean(dim=1, keepdim=True)
    col_mean = nn_k_col.mean(dim=0, keepdim=True)

    b = (row_mean + col_mean) * 0.5 + eps

    if variant == "distance":
        return a - b

    return a / b


def layer_accuracy(score: Tensor) -> float:
    device = score.device
    layer_predictions = score.argmax(dim=1)
    accuracy = (layer_predictions == torch.arange(score.size(0), device=device)).float()
    return accuracy.mean().item()


@torch.no_grad()
def layer_sentence_representation(
    model, tokenizer, sentences: list[str], batch_size: int = 32, device=None
) -> dict[int, Tensor]:
    temp: dict[int, list[Tensor]] = {}

    for start in range(0, len(sentences), batch_size):
        end = start + batch_size
        batch_sentences = sentences[start:end]

        tokens = tokenizer(batch_sentences, return_tensors="pt", padding=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        hidden_states = model(**tokens, output_hidden_states=True, use_cache=False).hidden_states

        for i, hidden_state in enumerate(hidden_states):
            mean_hidden = mean_hidden_state(hidden_states=hidden_state, attention_mask=tokens["attention_mask"])
            if i not in temp:
                temp[i] = []
            temp[i].append(mean_hidden)

        del hidden_states, tokens
        torch.cuda.empty_cache()

    ret: dict[int, Tensor] = {i: torch.cat(temp[i], dim=0) for i in temp.keys()}
    return ret


def mean_hidden_state(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    attention_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    unpad = hidden_states * attention_mask
    sum = unpad.sum(dim=1)
    k = attention_mask.sum(dim=1)
    mean = sum / k
    return mean


def generate_json_artifact(
    model_path: str,
    layer_acc: dict[int, float],
    base_lang: str,
    target_lang: str,
    save_dir: str,
):
    artifact = {
        "model_path": model_path,
        "base_lang": base_lang,
        "target_lang": target_lang,
        "timestamp": datetime.now().isoformat(),
        "layer_accuracy": sorted(layer_acc.items()),
    }

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    model_name = Path(model_path).parent.name
    checkpoint = Path(model_path).name
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(save_dir) / f"{timestamp_str}_{model_name}_{checkpoint}_{base_lang}-{target_lang}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=4, ensure_ascii=False)

    print(f"Artifact saved to {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Sentence retrieval task")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or a folder contains checkpoints")
    parser.add_argument("--base", type=str, required=True, help="Base language ISO code (e.g., eng)")
    parser.add_argument("--target", type=str, required=True, help="Target language ISO code (e.g., gle)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda or cpu)")
    parser.add_argument("--json-artifact-out", type=str, default="json_artifacts", help="Path to JSON accuracy artifacts")
    parser.add_argument("--margin-variant", type=str, default="ratio", help="Margin variant for score")
    args = parser.parse_args()
    
    print(f"Using margin variant {args.margin_variant}")

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load data
    print(f"Loading {args.base} and {args.target} FLORES datasets...")
    base_data = load_flores(args.base)
    target_data = load_flores(args.target)

    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model.eval()

    # Compute layer embeddings
    print(f"Computing layer embeddings for {args.base}...")
    base_ret = layer_sentence_representation(model, tokenizer, base_data, device=device)
    print(f"Computing layer embeddings for {args.target}...")
    target_ret = layer_sentence_representation(model, tokenizer, target_data, device=device)

    print(f"Computing cosine similarity between {args.target} and {args.base}...")
    layer_acc = {}
    for layer_index in base_ret.keys():
        score = margin_based_scoring(base_ret[layer_index], target_ret[layer_index], variant=args.margin_variant)
        layer_acc[layer_index] = layer_accuracy(score)

    generate_json_artifact(args.model_path, layer_acc, args.base, args.target, args.json_artifact_out)


if __name__ == "__main__":
    main()
