# aihub

A curated collection of local AI models with an idempotent downloader script.
Based on the research in [docs/](docs/) covering the open-model ecosystem as of 2025-2026.

---

## Prerequisites

- Python 3.9+
- Sufficient disk space (models range from ~1 GB to 43 GB each)
- GPU/CPU meeting VRAM requirements listed in `models.yaml`

---

## Installation

```bash
pip install -r requirements.txt
```

For faster downloads (optional, Rust-based):

```bash
pip install hf_transfer
```

---

## Configuration

1. Copy the credentials template:

```bash
cp .env.example .env
```

2. Edit `.env` and fill in your HuggingFace token (required for gated models like Llama):

```
HF_TOKEN=hf_your_token_here
HF_HUB_ENABLE_HF_TRANSFER=1   # optional: enables fast Rust-based downloads
```

Your HF token is available at: <https://huggingface.co/settings/tokens>

---

## Downloading Models

### List available models

```bash
python scripts/download_models.py --list
```

### Download all enabled models

```bash
python scripts/download_models.py
```

### Dry-run (validate model list without downloading)

```bash
python scripts/download_models.py --dry-run
```

### Download a specific model

```bash
python scripts/download_models.py --model Phi-4
python scripts/download_models.py --model Qwen2.5
```

### Download models by tag

```bash
python scripts/download_models.py --tag russian
python scripts/download_models.py --tag embeddings
python scripts/download_models.py --tag reasoning
```

### Force re-download (bypass ETag cache)

```bash
python scripts/download_models.py --force
python scripts/download_models.py --force --model bge-m3
```

### Use a different model list

```bash
python scripts/download_models.py --config my-selection.yaml
```

Models are saved to `./models/` (gitignored) in subdirectories by category.

---

## Idempotency

The downloader is fully idempotent:

- **Skip policy (default: `etag`)**: On each run the script checks the remote ETag via
  an HTTP HEAD request (no data transfer). If the local file's ETag matches, the download
  is skipped. Outdated files are re-downloaded automatically.
- **ETag sidecar files**: Each model file gets a `.etag` companion file storing the
  last-known ETag (e.g., `models/llm/qwen/Qwen2.5-14B-Instruct-Q4_K_M.gguf.etag`).
- **Atomic downloads**: `hf_hub_download()` writes to a temp file and renames on success,
  so interrupted downloads leave no partial files.

Change the policy in `models.yaml`:

```yaml
settings:
  update_policy: etag   # etag | skip | always
```

---

## Model List (`models.yaml`)

Edit `models.yaml` to control which models are downloaded:

```yaml
settings:
  models_dir: ./models       # local storage root
  default_quant: Q4_K_M     # informational label
  update_policy: etag        # etag | skip | always

models:
  - repo_id: bartowski/Qwen2.5-14B-Instruct-GGUF
    filename: Qwen2.5-14B-Instruct-Q4_K_M.gguf
    dest_dir: llm/qwen
    enabled: true             # set false to skip without removing the entry
    gated: false              # true = requires HF_TOKEN + license acceptance
    tags: [llm, chat, russian, 14b]
    vram_gb: 9
    description: "Qwen 2.5 14B — Alibaba, Apache 2.0"
```

### Fields

| Field         | Required | Description |
|---------------|----------|-------------|
| `repo_id`     | Yes      | HuggingFace `owner/repo` identifier |
| `filename`    | Yes      | Exact filename inside the repo (usually `*.gguf`) |
| `dest_dir`    | No       | Subdirectory inside `models/` (default: `misc`) |
| `enabled`     | No       | `false` to skip (default: `true`) |
| `gated`       | No       | `true` if model requires license acceptance + HF_TOKEN |
| `tags`        | No       | List of labels for `--tag` filtering |
| `vram_gb`     | No       | Minimum VRAM required (informational) |
| `description` | No       | Human-readable label |

---

## Repository Layout

```
aihub/
├── models.yaml              # Model download list (edit this to add/remove models)
├── requirements.txt         # Python dependencies
├── .env.example             # Credentials template (copy to .env)
├── .env                     # Your credentials (gitignored)
├── scripts/
│   └── download_models.py   # Main downloader script
├── docs/
│   └── *.md                 # Research documents
└── models/                  # Downloaded model files (gitignored)
    ├── llm/
    │   ├── llama/
    │   ├── qwen/
    │   ├── deepseek/
    │   ├── phi/
    │   ├── russian/
    │   └── code/
    ├── embeddings/
    └── image_gen/
```

---

## Included Models (default selection)

See `models.yaml` for the full annotated list. Default enabled models:

| Model | Category | VRAM | License |
|-------|----------|------|---------|
| Llama 3.1 8B Instruct Q4_K_M | LLM chat | 5 GB | Llama Community |
| Qwen 2.5 14B Instruct Q4_K_M | LLM chat | 9 GB | Apache 2.0 |
| DeepSeek R1 Distill Qwen 14B Q4_K_M | LLM reasoning | 10 GB | MIT / DeepSeek |
| Phi-4 14B Q4_K_M | LLM reasoning+code | 8 GB | MIT |
| Saiga Llama3 8B Q4_K | LLM Russian | 5 GB | Apache 2.0 |
| T-Lite 7B Q4_K_M | LLM Russian | 5 GB | Apache 2.0 |
| Qwen2.5-Coder 14B Q4_K_M | LLM code | 9 GB | Apache 2.0 |
| BGE-M3 Q8_0 | Embeddings RAG | 1 GB | Apache 2.0 |
| Nomic Embed Text v1.5 Q8_0 | Embeddings RAG | <1 GB | Apache 2.0 |

Disabled by default (large, require ≥20 GB VRAM): Qwen 2.5 32B, DeepSeek R1 32B,
Qwen2.5-Coder 32B, FLUX.1-schnell, SDXL.

---

## Security Notes

- `models/` and `.env` are gitignored — model files and credentials are never committed.
- `*.etag` sidecar files are also gitignored.
- Commit `.env.example` (with placeholder values) — never commit `.env`.

---

## Contributing

1. Add new models to `models.yaml` following the existing format.
2. Test with `--dry-run` before a full download.
3. Use `--list` to verify the model appears with correct metadata.
