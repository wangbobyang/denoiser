#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch as th

from audio import WaveReader, parse_scripts, read_wav, write_wav
from model_SC_CHM_Fusion import MS_SL2_split_model as OriginModel
from model_SC_CHM_Fusion_temporal_gate import MS_SL2_split_model as TemporalGateModel


DEFAULT_NNET_CONF = {
    "L": 16,
    "N": 512,
    "X": 4,
    "R": 1,
    "B": 256,
    "Sc": 256,
    "H": 512,
    "P": 3,
    "norm": "gLN",
    "num_spks": 2,
    "non_linear": "sigmoid",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_data_from_checkpoint(checkpoint):
    checkpoint_dir = Path(checkpoint).resolve().parent
    data_json = checkpoint_dir / "data.json"
    if not data_json.exists():
        return None, None, 16000
    data = load_json(data_json)
    dev = data.get("dev", {})
    return dev.get("mix_scp"), dev.get("ref_scp"), int(dev.get("sample_rate", 16000))


def load_nnet_conf(checkpoint):
    mdl_json = Path(checkpoint).resolve().parent / "mdl.json"
    if mdl_json.exists():
        return load_json(mdl_json)
    return dict(DEFAULT_NNET_CONF)


def load_checkpoint_state(checkpoint, device):
    cpt = th.load(checkpoint, map_location=device)
    if isinstance(cpt, dict) and "model_state_dict" in cpt:
        return cpt["model_state_dict"]
    return cpt


def build_model(model_type, checkpoint, device):
    conf = load_nnet_conf(checkpoint)
    if model_type == "origin":
        model = OriginModel(**conf)
    elif model_type == "temporal_gate":
        model = TemporalGateModel(**conf)
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))
    state = load_checkpoint_state(checkpoint, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[{}] missing keys: {}".format(model_type, len(missing)))
    if unexpected:
        print("[{}] unexpected keys: {}".format(model_type, len(unexpected)))
    model.to(device)
    model.eval()
    return model


def safe_key(key):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in key)


def to_1d_float(wav):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav[0]
    return wav


def run_model(model, mix, device):
    with th.no_grad():
        x = th.from_numpy(mix.astype(np.float32)).unsqueeze(0).to(device)
        outs = model(x)
        enhanced = []
        for out in outs:
            arr = out.squeeze(0).detach().cpu().numpy()
            enhanced.append(np.clip(arr, -1.0, 1.0))
        return enhanced


def sisnr_db(estimate, reference, eps=1e-8):
    n = min(len(estimate), len(reference))
    estimate = estimate[:n].astype(np.float64)
    reference = reference[:n].astype(np.float64)
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    proj = np.sum(estimate * reference) * reference / (np.sum(reference ** 2) + eps)
    noise = estimate - proj
    ratio = (np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps)
    return 10.0 * np.log10(ratio + eps)


def best_pit_sisnr(outputs, refs):
    if not refs:
        return None
    if len(outputs) != len(refs):
        return None
    if len(outputs) == 1:
        return sisnr_db(outputs[0], refs[0])
    scores = []
    for order in [(0, 1), (1, 0)]:
        scores.append(np.mean([sisnr_db(outputs[i], refs[order[i]]) for i in range(2)]))
    return max(scores)


def parse_run_specs(run_specs):
    runs = []
    for spec in run_specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Invalid --run '{}'. Use name:model_type:checkpoint_path".format(spec)
            )
        name, model_type, checkpoint = parts
        if model_type not in ("origin", "temporal_gate"):
            raise ValueError(
                "Invalid model type '{}'. Use origin or temporal_gate".format(model_type)
            )
        runs.append({
            "name": safe_key(name),
            "model_type": model_type,
            "checkpoint": checkpoint,
        })
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Generate enhanced wav files from two checkpoints for listening/comparison."
    )
    parser.add_argument(
        "--checkpoint1",
        default=r"C:\Users\user\Desktop\教材\自然語言\checkpoint\run1\best.pt.tar",
        help="First checkpoint, usually origin model.",
    )
    parser.add_argument(
        "--checkpoint2",
        default=r"C:\Users\user\Desktop\教材\自然語言\checkpoint\run2\best.pt.tar",
        help="Second checkpoint, usually temporal gate model.",
    )
    parser.add_argument("--model1", default="origin", choices=["origin", "temporal_gate"])
    parser.add_argument("--model2", default="temporal_gate", choices=["origin", "temporal_gate"])
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Add one run as name:model_type:checkpoint_path. "
            "Can be repeated, e.g. --run run1:origin:C:\\...\\run1\\best.pt.tar"
        ),
    )
    parser.add_argument("--mix-scp", default="", help="mix.scp to enhance. Defaults to checkpoint1 dev mix.scp.")
    parser.add_argument(
        "--ref-scp",
        nargs="*",
        default=None,
        help="Optional reference scp files, e.g. cv\\s1.scp cv\\s2.scp. Defaults to checkpoint1 dev refs if available.",
    )
    parser.add_argument("--sample-rate", type=int, default=0, help="Defaults to checkpoint1 data.json sample rate.")
    parser.add_argument("--max-files", type=int, default=10, help="Number of utterances to export; use 0 for all.")
    parser.add_argument("--out-dir", default="outputs_compare", help="Output directory.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    runs = parse_run_specs(args.run) if args.run else [
        {"name": "run1", "model_type": args.model1, "checkpoint": args.checkpoint1},
        {"name": "run2", "model_type": args.model2, "checkpoint": args.checkpoint2},
    ]

    inferred_mix_scp, inferred_ref_scp, inferred_sample_rate = infer_data_from_checkpoint(runs[0]["checkpoint"])
    mix_scp = args.mix_scp or inferred_mix_scp
    ref_scp = args.ref_scp if args.ref_scp is not None else inferred_ref_scp
    sample_rate = args.sample_rate or inferred_sample_rate

    if not mix_scp:
        raise RuntimeError("No mix.scp specified and checkpoint1/data.json was not found.")
    if not os.path.exists(mix_scp):
        raise FileNotFoundError("mix.scp not found: {}".format(mix_scp))

    device = th.device("cuda" if args.device == "auto" and th.cuda.is_available() else args.device)
    if args.device == "auto" and not th.cuda.is_available():
        device = th.device("cpu")
    print("Using device:", device)
    print("mix.scp:", mix_scp)

    models = []
    for run in runs:
        print("Loading {} ({}) from {}".format(run["name"], run["model_type"], run["checkpoint"]))
        models.append({
            "name": run["name"],
            "model": build_model(run["model_type"], run["checkpoint"], device),
        })
    mix_reader = WaveReader(mix_scp, sample_rate=sample_rate)

    ref_readers = []
    if ref_scp:
        ref_readers = [WaveReader(path, sample_rate=sample_rate) for path in ref_scp]

    out_dir = Path(args.out_dir)
    rows = []
    count = len(mix_reader) if args.max_files == 0 else min(args.max_files, len(mix_reader))

    for idx, key in enumerate(mix_reader.index_keys[:count], start=1):
        mix = to_1d_float(mix_reader[key])
        outputs_by_run = []
        for item in models:
            outputs_by_run.append({
                "name": item["name"],
                "outputs": run_model(item["model"], mix, device),
            })
        refs = [to_1d_float(reader[key]) for reader in ref_readers] if ref_readers else []

        utt = safe_key(key)
        mix_out = out_dir / "mix" / "{}.wav".format(utt)
        write_wav(str(mix_out), np.clip(mix, -1.0, 1.0), fs=sample_rate)

        row = {"key": key}
        for item in outputs_by_run:
            run_name = item["name"]
            for spk, wav in enumerate(item["outputs"], start=1):
                write_wav(str(out_dir / run_name / "s{}".format(spk) / "{}.wav".format(utt)), wav, fs=sample_rate)
            row["{}_pit_sisnr_db".format(run_name)] = best_pit_sisnr(item["outputs"], refs)
        rows.append(row)
        print("[{}/{}] exported {}".format(idx, count, key))

    csv_path = out_dir / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["key"] + ["{}_pit_sisnr_db".format(run["name"]) for run in runs]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print("Listen here:")
    print("  mix :", out_dir / "mix")
    for run in runs:
        print("  {}: {}".format(run["name"], out_dir / run["name"]))
    print("Metrics:", csv_path)


if __name__ == "__main__":
    main()
