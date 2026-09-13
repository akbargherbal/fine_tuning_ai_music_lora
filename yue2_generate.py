#!/usr/bin/env python3
"""Quick YuE2-3B song generator (Arabic / Suno-manifest workflow).

Canonical script for this project. Turn a track from workspace_manifest.json
(Suno style + lyrics, sent to YuE2 verbatim, unmodified) into one YuE2 song
with the L4-friendly default settings.

Examples
--------
Generate from a manifest track (style/lyrics sent as-is):
    python yue2_generate.py --match 07- --seed 831001 --out outputs/badr

Higher quality (slower), or max quality with an automatic legacy-VAE decode:
    python yue2_generate.py --match 07- --seed 831001 --quality high --out outputs/badr_high
    python yue2_generate.py --match 07- --seed 831001 --quality max --out outputs/badr_max

Override style/lyrics from files (skip the manifest):
    python yue2_generate.py --style-file style.txt --lyrics-file lyrics.txt --out outputs/custom

Decode an already-generated song with the legacy VAE (cheap quality A/B):
    python yue2_generate.py --redecode outputs/badr/artifacts/latent.npy --vae m-a-p/YuE2-Vae-legacy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

MANIFEST = Path("/content/YuE2-3B/workspace_manifest.json")
MODEL = "/content/models/YuE2-3B"
VAE = "/content/models/YuE2-Vae"
VAE_LEGACY = "/content/models/YuE2-Vae-legacy"

# Quality presets: higher ode_steps = more detail, more generation time.
# max also decodes with the legacy VAE (near-free extra ~10s, scored higher
# on musicality in the model's own benchmark) so you get both to compare.
QUALITY_PRESETS = {
    "standard": {"ode_steps": 32},
    "high": {"ode_steps": 48},
    "max": {"ode_steps": 64, "also_legacy_vae": True},
}


def pick_track(manifest: Path, match: str) -> dict:
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for t in data["tracks"]:
        if t.get("assigned_filename", "").startswith(match) or match in t.get("original_title", ""):
            return t
    raise SystemExit(f"No track matching {match!r} in {manifest}")


def redecode(latents_path: str, vae: str) -> None:
    import numpy as np
    from yue2 import YuE2Pipeline

    pipe = YuE2Pipeline.from_pretrained(MODEL, vae=VAE, device="cuda")
    latents = np.load(latents_path)
    audio = pipe.decode(latents, vae=vae)
    out = Path(latents_path).with_name(Path(latents_path).stem + f"_{Path(vae).name}.flac")
    import soundfile as sf
    sf.write(out, audio, 48000, subtype="PCM_24")
    print(f"wrote {out}")
    pipe.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", default="07-", help="manifest filename/title prefix (default 07-)")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--style-file", help="override: style text file")
    ap.add_argument("--lyrics-file", help="override: lyrics text file")
    ap.add_argument("--seed", type=int, default=831001)
    ap.add_argument("--cfg", type=float, default=1.2, help="text-guidance scale (not a negative prompt)")
    ap.add_argument("--cot", default="full", choices=["full", "melody", "off"])
    ap.add_argument("--quality", default="standard", choices=list(QUALITY_PRESETS),
                     help="standard=fast (ode_steps=32), high=ode_steps=48, "
                          "max=ode_steps=64 + auto legacy-VAE decode")
    ap.add_argument("--ode-steps", type=int, default=None,
                     help="NAR flow-matching steps; overrides --quality's default if set")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--vae", default=VAE, help="VAE path or HF repo (e.g. m-a-p/YuE2-Vae-legacy)")
    ap.add_argument("--vae-legacy", default=VAE_LEGACY, help="legacy VAE path, used by --quality max")
    ap.add_argument("--out", default="outputs/song")
    ap.add_argument("--redecode", metavar="LATENTS.NPY", help="re-decode existing latents instead of generating")
    args = ap.parse_args()

    preset = QUALITY_PRESETS[args.quality]
    ode_steps = args.ode_steps if args.ode_steps is not None else preset["ode_steps"]
    also_legacy_vae = preset.get("also_legacy_vae", False)

    if args.redecode:
        redecode(args.redecode, args.vae)
        return

    if args.style_file and args.lyrics_file:
        style = Path(args.style_file).read_text(encoding="utf-8").strip()
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
        title = "custom"
    else:
        track = pick_track(args.manifest, args.match)
        title = track["original_title"]
        style = track["styles"]
        lyrics = track["lyrics"]
        print("EXCLUDE (not usable by YuE2):", track.get("exclude_styles", ""))

    print("TITLE  :", title)
    print("STYLE  :", style)
    print("SEED   :", args.seed, "| CFG:", args.cfg, "| CoT:", args.cot,
          "| QUALITY:", args.quality, "(ode_steps=" + str(ode_steps) + ")")
    print("LYRICS :\n" + lyrics)
    sys.stdout.flush()

    import torch
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    t0 = time.perf_counter()
    config = GenerationConfig(ode_steps=ode_steps)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda", generation_config=config)
    print(f"\n[loaded pipeline in {time.perf_counter() - t0:.1f}s]", flush=True)

    t1 = time.perf_counter()
    song = pipe(style=style, lyrics=lyrics, cot=args.cot, seed=args.seed, cfg_scale=args.cfg)
    gen = time.perf_counter() - t1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    song.save(str(out / "song.flac"))
    song.save_artifacts(str(out / "artifacts"))

    if also_legacy_vae:
        t2 = time.perf_counter()
        import soundfile as sf
        legacy_audio = pipe.decode(song.latents, vae=args.vae_legacy)
        sf.write(out / "song_legacy.flac", legacy_audio, 48000, subtype="PCM_24")
        print(f"[legacy VAE decode in {time.perf_counter() - t2:.1f}s] -> {out / 'song_legacy.flac'}")

    meta = {
        "title": title, "style": style,
        "seed": args.seed, "cfg_scale": args.cfg, "cot": args.cot,
        "quality": args.quality, "ode_steps": ode_steps, "legacy_vae_decoded": also_legacy_vae,
        "generation_seconds": gen, "timing": song.timing,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nDONE in {gen:.1f}s -> {out / 'song.flac'}")
    print("peak VRAM GiB:", round(meta["peak_vram_gib"], 2))
    pipe.close()


if __name__ == "__main__":
    main()
