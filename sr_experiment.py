"""
Sentence retrieval task accuracy on FLORES-PRO dataset

Ref: https://aclanthology.org/2024.loresmt-1.20.pdf
"""

import argparse
from datetime import datetime
import json
import os

from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import Tensor

import hashlib

from datasets import load_dataset, concatenate_datasets
from datasets.arrow_dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map="auto") -> AutoModelForCausalLM:
    """Load a model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model path {model_path} does not exist.")
    print(f"Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def load_flores(dataset: str) -> list[str]:
    FLORES = "openlanguagedata/flores_plus"

    ds = load_dataset(FLORES, dataset)

    ds = concatenate_datasets([ds["dev"], ds["devtest"]])
    #ds = ds["devtest"]

    return [d["text"] for d in ds]


def cosine_similarity(first: Tensor, second: Tensor) -> Tensor:
    """
    Compute cosine similarity between 2 sets of sentences
    Output will be a matrix where
    - Each row is cosine sim between first sentence and all other second sentences
    first: [sentence_count, d_model]
    second: [sentence_count, d_model]

    out: [sentence_count, sentence_count]
    """
    first_normalized: Tensor = first / torch.linalg.norm(
        first, ord=2, dim=1, keepdim=True
    )
    second_normalized: Tensor = second / torch.linalg.norm(
        second, ord=2, dim=1, keepdim=True
    )

    return first_normalized @ second_normalized.T


def layer_accuracy(cos_sim: Tensor) -> float:
    """
    Calculate the layer accuracy for the sentence retrieval task
    cos_sim: [sentence_count, sentence_count]

    if the argmax for each row is the index for the row itself then we will accum 1 to the total
    Then divide by sentence_count
    """

    # [sentence_count]
    layer_predictions = cos_sim.argmax(dim=1)

    # [sentence_count]
    accuracy = (layer_predictions == torch.arange(cos_sim.size(0))).float()

    # scalar
    return accuracy.mean().item()


@torch.no_grad()
def layer_sentence_representation(
    model, tokenizer, sentences: list[str], batch_size: int = 32
) -> dict[int, list[Tensor]]:
    """
    Return the mean hidden state of each sentence for each layer
    Note: Qwen has 24 transformer layers and 1 input embedding layers (25 in total)

    Update 1: Add batching hopefully this solves OOM
    Update 2: It solved OOM I think

    model: The model to use
    tokenizer: The tokenizer to use
    sentences: List of sentences

    {
        layer_index: [sentence_count, d_model]
    }
    """

    temp: dict[int, list[Tensor]] = {}

    tokens = tokenizer(sentences, return_tensors="pt", padding=True)

    for start in range(0, len(sentences), batch_size):
        end = start + batch_size
        batch_sentences = sentences[start:end]

        tokens = tokenizer(batch_sentences, return_tensors="pt", padding=True)

        # Update 1: Use cache False so won't OOM GPU according to the prof GPT
        # Update 2: OOM still occurs (GPT lies)
        # tuple(layer_index, Tensor(batch_size, d_model))
        hidden_states = model(
            **tokens, output_hidden_states=True, use_cache=False
        ).hidden_states

        for i, hidden_state in enumerate(hidden_states):
            mean_hidden = mean_hidden_state(
                hidden_states=hidden_state, attention_mask=tokens["attention_mask"]
            )

            if i not in temp:
                temp[i] = []

            temp[i].append(mean_hidden)

        del hidden_states, tokens
        torch.cuda.empty_cache()

    ret: dict[int, Tensor] = {}

    for i in temp.keys():
        # (sentence_count, d_model)
        ret[i] = torch.cat(temp[i], dim=0)

    return ret


def mean_hidden_state(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Take average of all token embeddings for a sentence hidden state
    Create a single vector representation for that sentence
    We do things in batch

    Params:
    hidden_states: [batch, seq, d_model]
    attention_mask: [batch, seq]

    Returns:
    mean: [batch, d_model]
    """
    # Unsqueeze to [batch, seq, 1]
    attention_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)

    # Remove pad [batch, seq, d_model]
    unpad = hidden_states * attention_mask

    # Sum over seq [batch, d_model]
    sum = unpad.sum(dim=1)

    # Sum over seq [batch, 1]
    k = attention_mask.sum(dim=1)

    # [batch, d_model]
    mean = sum / k
    return mean


def generate_json_artifact(
    model_path: str,
    layer_acc: dict[int, float],
    base_lang: str,
    target_lang: str,
    save_dir: str,
):
    """
    Generate a JSON artifact summarizing sentence retrieval accuracy per layer.

    Args:
        model_path (str): Path or name of the model used.
        layer_acc (dict[int, float]): Accuracy for each layer, e.g., {0: 0.12, 1: 0.34, ...}.
        base_lang (str): Source language code.
        target_lang (str): Target language code.
        save_dir (str): Directory to save the JSON artifact.
    """

    artifact = {
        "model_path": model_path,
        "base_lang": base_lang,
        "target_lang": target_lang,
        "timestamp": datetime.now().isoformat(),
        "layer_accuracy": sorted(layer_acc.items()),
    }

    Path(save_dir).mkdir(exist_ok=True, parents=True)

    # We are using the convention <model_name>/<checkpoint>
    model_name = Path(model_path).parent.name
    checkpoint = Path(model_path).name

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(save_dir) / f"{timestamp_str}_{model_name}_{checkpoint}_{base_lang}-{target_lang}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=4, ensure_ascii=False)

    print(f"Artifact saved to {json_path}")
    return json_path

def main():
    parser = argparse.ArgumentParser(
        description="Sentence retrieval task"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--base", type=str, required=True, help="Base language ISO code (e.g., eng)"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Target language ISO code (e.g., gle)"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--json-artifact-out",
        type=str,
        default="json_artifacts",
        help="Path to JSON accuracy artifacts",
    )
    parser.add_argument(
        "--plot-artifact-out",
        type=str,
        default="plot_artifacts",
        help="Path to plot artifact results",
    )
    args = parser.parse_args()

    print(f"Loading {args.base} and {args.target} FLORES datasets...")
    base_data = load_flores(args.base)
    target_data = load_flores(args.target)

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()

    print(f"Computing layer embeddings for {args.base}...")
    base_ret = layer_sentence_representation(model, tokenizer, base_data)
    print(f"Computing layer embeddings for {args.target}...")
    target_ret = layer_sentence_representation(model, tokenizer, target_data)

    print(f"Computing cosine similarity between {args.target} and {args.base}")

    layer_acc = {}
    # Loop through each layer

    for layer_index, _ in base_ret.items():
        cos_sim = cosine_similarity(base_ret[layer_index], target_ret[layer_index])
        layer_acc[layer_index] = layer_accuracy(cos_sim)

    # for i, _ in enumerate(base_ret):
    #     cos_sim = cosine_similarity(base_ret[i], target_ret[i])
    #     layer_acc[i] = layer_accuracy(cos_sim)

    # Generate artifact json file
    generate_json_artifact(
        args.model_path,
        layer_acc,
        args.base,
        args.target,
        args.json_artifact_out
    )


if __name__ == "__main__":
    main()
