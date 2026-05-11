# FedAER

Code for Resource-Constrained Personalized Federated Relation Classification with Entity and Relation Representation Alignment. Our paper is currently under review, and we will release the full code upon acceptance.

This repository is organized as a **reproducible codebase**, not as an experiment dump. Training logs, temporary launcher scripts, generated result tables, local IDE files, and partitioned datasets are intentionally excluded from version control.

## What Is Included

- training and evaluation code for `FedAER`
- baseline implementations used in the paper
- model definitions and local PEFT/transformers modifications
- dataset loading and partition preprocessing scripts
- dependency specification

## What Is Not Included

- training logs and launcher outputs
- ad-hoc `.sh` batch scripts used on specific servers
- generated experiment tables and spreadsheets
- generated partition files under `tasks/*/FL_data/`
- raw dataset files under `tasks/*/raw_data/`
- local IDE configuration and temporary backups

## Environment

Recommended environment:

- Python `3.8`
- PyTorch `1.9.1`
- CUDA `11.1`

Install dependencies with:

```bash
pip install -r requirements.txt
```

The project uses the local modified transformers package under:

```bash
model/adapter_src/src
```

Before running experiments, add it to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/model/adapter_src/src:$PYTHONPATH"
```

## Data Preparation

This repository does not ship raw datasets or generated federated partitions.

You need to prepare the raw data files under:

- `tasks/fewrel/raw_data/`
- `tasks/nyt/raw_data/`
- `tasks/tacred/raw_data/`

Then generate federated partitions with the provided Python scripts:

- shard-based label skew:
  - `tasks/fewrel/new_niid_per.py`
  - `tasks/nyt/new_niid_per.py`
  - `tasks/tacred/new_niid_per.py`
- Dirichlet label skew:
  - `tasks/fewrel/niid_label_nna_per.py`
  - `tasks/nyt/niid_label_nna_per.py`
  - `tasks/tacred/niid_label_nna_per.py`
- quantity skew:
  - `tasks/fewrel/niid_quantity_nna_per.py`
  - `tasks/nyt/niid_quantity_nna_per.py`
  - `tasks/tacred/niid_quantity_nna_per.py`

Example:

```bash
python tasks/fewrel/new_niid_per.py --num_clients 10 --alpha 2
python tasks/nyt/niid_label_nna_per.py --num_clients 10 --alpha 0.1
python tasks/tacred/niid_quantity_nna_per.py --num_clients 10 --alpha 1
```

The generated files are expected under `tasks/*/FL_data/` and are intentionally ignored by Git.

## Running

The main entry is:

```bash
python main.py
```

Important arguments:

- `--algorithm`: `FedAER`, `FedAvg`, `FedProx`, `MOON`, `Per_FedAvg`, `Ditto`, `pFedMe`, `FedFomo`, `FedPAC`
- `--dataset_name`: `fewrel`, `nyt`, `tacred`
- `--niid`: `niid_label`, `dir_label`, `niid_quantity`
- `--alpha`: partition parameter
- `--lora`, `--adapter`, `--bitfit`, `--prefix`: PEFT backbone choice
- `--entity_loss`, `--proto_loss`: enable the two FedAER alignment terms
- `--model_name_or_path`: local path or model identifier for the PLM backbone

Example commands:

```bash
python main.py \
  --algorithm FedAER \
  --dataset_name fewrel \
  --niid niid_label \
  --alpha 2 \
  --lora \
  --entity_loss \
  --proto_loss \
  --train_epochs 5 \
  --communication_rounds 50 \
  --batch_size 32 \
  --sample_rate 1.0 \
  --model_name_or_path /path/to/bert-base-uncased
```

```bash
python main.py \
  --algorithm FedAvg \
  --dataset_name tacred \
  --niid niid_label \
  --alpha 4 \
  --lora \
  --train_epochs 5 \
  --communication_rounds 50 \
  --batch_size 32 \
  --sample_rate 1.0 \
  --model_name_or_path /path/to/bert-base-uncased
```

## Repository Layout

```text
FedPrompt/
├── baselines/                  # baseline FL/PFL implementations
├── model/                      # model definitions and local transformers fork
├── tasks/                      # dataset loader and partition preprocessing
├── training/                   # training loops
├── get_trainer.py              # FedAER trainer entry
├── main.py                     # main experiment entry
├── requirements.txt
└── README.md
```

## Notes

- The default `main.py` path for `--model_name_or_path` is server-specific. Override it in your own environment.
- This repository intentionally excludes server-side experiment orchestration artifacts. Reproducibility is provided through source code, dependency versions, preprocessing scripts, and direct command-line examples.
