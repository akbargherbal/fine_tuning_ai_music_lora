# YuE2-3B — Arabic Music Generation Project

Testing the open-source **[`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B)**
music model against an existing Suno project, to see how well a Suno-style
prompt/lyrics workflow transfers to this model **as-is** — no music-theory
editing, no prompt cleaning. Runs on a Google Colab **NVIDIA L4 (24 GB)** instance.

## How it works, in short

YuE2 generation happens in two stages:

1. **Compose** — the model reads your `style` tags and `lyrics` and produces
   a compressed internal representation of the song ("latents"). This is the
   slow, expensive step (several minutes).
2. **Decode** — a separate component, the **VAE**, turns those latents into
   an actual audio waveform. This step is cheap (~10–15 seconds), and because
   it's separate from composing, you can re-run *just* this step with a
   different VAE to get an alternate-sounding render of the same song without
   regenerating it. Two VAEs are available: the default `YuE2-Vae`, and
   `YuE2-Vae-legacy`, which scored higher on musicality in the model's own
   benchmark.

Everything is driven from `workspace_manifest.json` — a Suno export containing
your tracks' `styles` and `lyrics` — and `style`/`lyrics` are sent to YuE2
**exactly as they appear there**, Suno control syntax included. The point of
this project is to measure that transfer, not to adapt the input for the model.

## Repo layout

```
YuE2-3B/
├── README.md                 # this file
├── setup.sh                  # one-time environment + model setup
├── requirements-lock.txt     # exact package versions used
├── workspace_manifest.json   # Suno source data (tracks: styles + lyrics)
├── yue2_generate.py          # the main script — generate one song
├── yue2_batch.py             # best-of-N batch runner (pre-dates the current
│                              # simplified workflow — not yet updated to match
│                              # yue2_generate.py; on hold until the single-song
│                              # workflow is settled)
├── listen.py                 # Colab A/B audio player
├── logs/                     # run logs
└── outputs/                  # generated songs, one subfolder per run
```

## Environment

| | |
|---|---|
| GPU | NVIDIA L4, 24 GB class |
| Python | 3.13 |
| Key packages | `torch`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `yue2` |

The `yue2` package ships as a wheel with pinned dependency versions (see
below) — it isn't just `pip install yue2`.

## Setup

Run once per fresh environment (e.g. a new Colab session):

```bash
bash setup.sh
```

What it does:
1. Installs `huggingface-hub==0.36.2` and `transformers==4.57.6` — the `yue2`
   package needs this exact pair; Colab's default `transformers` (5.x) is
   incompatible with `huggingface-hub==0.36.2` and breaks on import.
2. Downloads and installs the `yue2` inference package itself, with
   `--no-deps` (so it doesn't try to downgrade Colab's newer `torch`, which
   works fine despite the wheel declaring an older pin).
3. Downloads the model weights to `/content/models/`: `YuE2-3B` (main model,
   ~7.3 GB), `YuE2-Vae` (~0.5 GB), and `YuE2-Vae-legacy`.
4. Prints a quick import check to confirm everything loaded.

**Hugging Face token:** all of the above are public repos, so a token is
never required — but if `HF_TOKEN` is set in your environment (or you're
authenticated another way), downloads are faster due to higher rate limits.
If `HF_TOKEN` isn't already set, `setup.sh` will prompt you once for it
(press Enter to skip); if it's not set and there's no interactive terminal,
it skips the prompt automatically. If a token turns out to be invalid or
expired, the script drops it and retries the download anonymously rather
than failing outright.

Model weights are large and not meant to be kept in backups — re-run
`setup.sh` after restoring the rest of the project to a fresh environment.

## Generating a song

```bash
python yue2_generate.py --match 07- --seed 831001 --out outputs/badr
```

`--match` picks a track from `workspace_manifest.json` by filename prefix or
title substring. See **`yue2_generate_README.md`** for the full flag
reference — quality presets (`--quality standard|high|max`), overriding
style/lyrics with your own files, re-decoding existing latents with a
different VAE, and what gets written to the output folder.

## Current findings (as of the last research pass)

- The model runs comfortably on an L4: ~6 minutes for a 5-minute song, under
  10 GiB peak VRAM.
- Subjectively (by ear, n small): noticeably better than ACE-Step 1.5,
  approaching Suno 3.5 on this material.
- The model consistently defaults to Western minor keys regardless of the
  maqam requested in the style text — this held across every generation mode
  tested, so it appears to be a bias in the model itself rather than
  something fixable via prompt phrasing.
- YuE2 has no negative-prompt input, so anything Suno's `exclude_styles`
  asked to avoid (e.g. no oud/qanun/darbuka, no autotune) can't be enforced.
- Arabic is out-of-distribution for this model (its card lists English and
  Chinese); this sets a real quality ceiling, though results so far have
  still been considered promising.

## Licensing

`m-a-p/YuE2-3B` weights are **CC BY-NC 4.0** — non-commercial use only. Check
the model repo for third-party code licenses before any commercial use.
