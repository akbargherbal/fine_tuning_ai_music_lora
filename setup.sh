#!/usr/bin/env bash
# Reproducible YuE2-3B setup for Google Colab (L4 / 24GB class GPU).
# The only tricky part: the yue2-infer wheel pins an exact, older
# (transformers, huggingface-hub) pair. Newer huggingface-hub (>=1.0) breaks
# transformers 5.x, so we install the matching transformers 4.57.6 and keep
# Colab's torch 2.11 CUDA build instead of downgrading to the pinned 2.10.
set -euo pipefail

# --- HF_TOKEN: optional, only speeds up downloads via higher rate limits. ---
# These are all public repos, so a token is never required. If HF_TOKEN isn't
# already set in the environment, ask once (skipped automatically if there's
# no interactive terminal, e.g. running unattended). Anything entered here is
# used only for this session's downloads.
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -t 0 ]; then
        read -r -p "HF_TOKEN not set. Paste a Hugging Face token to speed up downloads (Enter to skip): " HF_TOKEN_INPUT
        if [ -n "$HF_TOKEN_INPUT" ]; then
            export HF_TOKEN="$HF_TOKEN_INPUT"
        fi
    else
        echo "HF_TOKEN not set (no interactive terminal) — continuing without it."
    fi
fi

# Downloads all repos are public, so a missing/invalid token should never fail
# the setup. Try the download; if it fails while a token is set, drop the
# token (it was probably wrong/expired) and retry once without it.
hf_download() {
    if hf download "$@"; then
        return 0
    fi
    if [ -n "${HF_TOKEN:-}" ]; then
        echo "Download failed with HF_TOKEN set (likely invalid/expired token) — retrying without it..."
        unset HF_TOKEN
        hf download "$@"
    else
        return 1
    fi
}

python -m pip install -q "huggingface-hub==0.36.2"
python -m pip install -q "transformers==4.57.6"

mkdir -p /content/YuE2-3B/pkg /content/models
hf_download m-a-p/YuE2-3B yue2_infer-0.1.5-py3-none-any.whl --local-dir /content/YuE2-3B/pkg
python -m pip install -q --no-deps /content/YuE2-3B/pkg/yue2_infer-0.1.5-py3-none-any.whl

hf_download m-a-p/YuE2-3B --local-dir /content/models/YuE2-3B
hf_download m-a-p/YuE2-Vae --local-dir /content/models/YuE2-Vae
hf_download m-a-p/YuE2-Vae-legacy --local-dir /content/models/YuE2-Vae-legacy

python -c "import torch, transformers, yue2; print('torch', torch.__version__, '| transformers', transformers.__version__, '| yue2 OK')"
