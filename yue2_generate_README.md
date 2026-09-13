# yue2_generate.py — Usage Reference

CLI for turning a Suno-sourced track (style + lyrics) into a generated song
via the **YuE2-3B** model. Written for a Colab-style environment with an L4 GPU.

`style`/`lyrics` are sent to YuE2 **verbatim** — exactly as they appear in the
manifest — with no reformatting. The goal of this script is to measure how
well a Suno-style prompt/lyrics pair transfers to this open-source model
as-is, not to adapt the input for it.

## Prerequisites

These are **not** installed or created by the script — they must already exist
in the environment before running it:

| Requirement | Notes |
|---|---|
| CUDA GPU | Script hardcodes `device="cuda"` |
| `yue2` Python package | Provides `YuE2Pipeline` and `yue2.protocol.GenerationConfig` |
| `torch`, `soundfile`, `numpy` | Standard deps, used for tensors / audio I/O |
| Model weights at `/content/models/YuE2-3B` | Or pass `--model <path>` |
| VAE weights at `/content/models/YuE2-Vae` | Or pass `--vae <path or HF repo>` |
| `workspace_manifest.json` at `/content/YuE2-3B/workspace_manifest.json` | JSON file with a `tracks` list (see below), or use `--style-file`/`--lyrics-file` instead |

### Manifest track shape
Each entry in `data["tracks"]` should look like:
```json
{
  "original_title": "...",
  "assigned_filename": "07-...",
  "styles": "genre: \"...\" vocals: \"...\" instrumentation: \"...\" mood: \"...\" production: \"...\"",
  "lyrics": "[Verse 1 | ...]\n...",
  "exclude_styles": "..."
}
```
`styles` and `lyrics` are passed to YuE2 exactly as stored here, Suno control
syntax and all (`[Verse 1 | ...]`, `///***///` separators, etc.) — the model
sees the same text you'd hand to Suno.

## Usage modes

**1. Generate from a manifest track**
```bash
python yue2_generate.py --match 07- --seed 831001 --out outputs/badr
```
`--match` selects the first track whose `assigned_filename` starts with the
string, or whose `original_title` contains it.

**2. Pick a quality preset instead of tuning knobs by hand**
```bash
python yue2_generate.py --match 07- --seed 831001 --quality high --out outputs/badr_high
python yue2_generate.py --match 07- --seed 831001 --quality max  --out outputs/badr_max
```
| Preset | What it sets | Trade-off |
|---|---|---|
| `standard` (default) | `ode_steps=32` | Fastest |
| `high` | `ode_steps=48` | ~40–60s slower per song, more detail |
| `max` | `ode_steps=64` + auto legacy-VAE decode | Same slowdown as `high`, plus a ~10s extra decode so you get `song.flac` **and** `song_legacy.flac` to A/B |

`ode_steps` is the refinement-pass count during audio synthesis — higher
means more detail at the cost of more generation time. If you pass
`--ode-steps` explicitly, it overrides whatever the preset would have set.
`--cfg` and `--cot` are deliberately **not** part of these presets — they're
separate trade-offs (prompt adherence, song-structure planning) that aren't
simply "more quality," so they stay independent flags you can combine with
any preset if you want to experiment.

**3. Bypass the manifest with your own files**
```bash
python yue2_generate.py --style-file style.txt --lyrics-file lyrics.txt --out outputs/custom
```

**4. Re-decode existing latents with a different VAE**
```bash
python yue2_generate.py --redecode outputs/badr/artifacts/latent.npy --vae m-a-p/YuE2-Vae-legacy
```
Useful for cheaply A/B-testing VAE quality without re-running the full model.

## All flags

| Flag | Default | Meaning |
|---|---|---|
| `--match` | `07-` | Manifest filename/title prefix to select a track |
| `--manifest` | `/content/YuE2-3B/workspace_manifest.json` | Path to manifest JSON |
| `--style-file` | — | Override: plain-text style file (must pair with `--lyrics-file`) |
| `--lyrics-file` | — | Override: plain-text lyrics file |
| `--seed` | `831001` | Generation seed |
| `--cfg` | `1.2` | Text-guidance scale (not a negative prompt) |
| `--cot` | `full` | Chain-of-thought mode: `full`, `melody`, or `off` |
| `--quality` | `standard` | Preset: `standard` (fast), `high`, or `max` (also decodes with the legacy VAE) — see table above |
| `--ode-steps` | preset-dependent | NAR flow-matching steps; overrides `--quality`'s value if set explicitly |
| `--model` | `/content/models/YuE2-3B` | Model path |
| `--vae` | `/content/models/YuE2-Vae` | VAE path or HF repo |
| `--vae-legacy` | `/content/models/YuE2-Vae-legacy` | Legacy VAE path, used automatically by `--quality max` |
| `--out` | `outputs/song` | Output directory |
| `--redecode` | — | Path to a `.npy` latents file; skips generation and just re-decodes |

## Output

Written to the `--out` directory:
- `song.flac` — generated audio
- `song_legacy.flac` — only with `--quality max`: the same song decoded with the legacy VAE, for comparison
- `artifacts/` — intermediate generation artifacts, including the latents
  file usable later with `--redecode`
- `run_meta.json` — title, style, seed, cfg_scale, cot, quality preset, ode_steps,
  generation time in seconds, full timing breakdown, and peak VRAM (GiB)

## Notes / things to double check before running

- Nothing in the script installs `yue2` or downloads model weights — confirm
  those are already present in your environment (see the project's `setup.sh`).
- Paths default to `/content/...` (Colab convention). If running elsewhere,
  use `--manifest`, `--model`, and `--vae` to point at real paths.
- `--style-file`/`--lyrics-file` mode only activates if **both** are given —
  if only one is set, the script falls through to manifest lookup instead.
- `exclude_styles` in the manifest is printed for reference but **not sent**
  to YuE2 — the model has no negative-prompt input, so anything Suno was
  told to avoid can't be enforced here.

