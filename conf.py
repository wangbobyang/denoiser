from pathlib import Path

fs = 16000
chunk_len = 3  # (s)
chunk_size = chunk_len * fs
num_spks = 2

nnet_conf = {
    "L": 16,
    "N": 512,
    "X": 4,
    "R": 1,
    "B": 256,
    "Sc": 256,
    "H": 512,
    "P": 3,
    "norm": "gLN",  # BN, gLN, cLN
    "num_spks": num_spks,
    "non_linear": "sigmoid",
}

data_root = Path(r"C:\Users\user\Desktop\教材\自然語言\final_dataset\final_dataset")
train_dir = data_root / "tr"
dev_dir = data_root / "cv"

train_data = {
    "mix_scp": str(train_dir / "mix.scp"),
    "ref_scp": [str(train_dir / f"s{n}.scp") for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
}

dev_data = {
    "mix_scp": str(dev_dir / "mix.scp"),
    "ref_scp": [str(dev_dir / f"s{n}.scp") for n in range(1, 1 + num_spks)],
    "sample_rate": fs,
}

adam_kwargs = {
    "lr": 0.001,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200,
    "no_impr": 100,
    "loss_mode": "sisnr",
}
