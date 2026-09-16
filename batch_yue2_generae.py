#!/usr/bin/env python3
"""Batch YuE2-3B song generator (Arabic / Suno-export workflow).

Run every track in a batch JSON file (a list of manifest-shaped entries such as
batch_01.json) through the same pipeline as yue2_generate.py. The model and VAE
are loaded once and reused for the whole batch, and each song is written to its
own subfolder under --out.

Examples
--------
    python batch_yue2_generae.py --batch batch_01.json --quality high --out outputs/batch_01
    python batch_yue2_generae.py --batch batch_01.json --limit 2 --seed 831001 --out outputs/try
    python batch_yue2_generae.py --batch batch_01.json --overwrite
    python batch_yue2_generae.py --batch batch_01.json --dry-run

The generation flags (--seed, --cfg, --cot, --quality, --ode-steps, --model,
--vae, --vae-legacy, --out) behave exactly as in yue2_generate.py; --batch
replaces the manifest/override track-selection flags.

Each batch entry is a manifest-shaped dict; only styles/lyrics are required:
    {
      "original_title": "...",
      "assigned_filename": "07-..._SONG_B.mp3",
      "styles": "<Suno style text, sent verbatim to YuE2>",
      "lyrics": "<Suno lyrics, sent verbatim to YuE2>",
      "seed": 831001,        # optional per-track override
      "cfg": 1.2,            # optional per-track override
      "cot": "full",         # optional per-track override
      "out": "custom_name"   # optional output subfolder override
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yue2_generate as solo


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="batch JSON file: a list of manifest-shaped track entries")
    solo.add_generation_args(ap)
    ap.set_defaults(out="outputs/batch")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N tracks")
    ap.add_argument("--overwrite", action="store_true", help="regenerate tracks whose song.flac already exists")
    ap.add_argument("--fail-fast", action="store_true", help="stop the whole batch on the first track that errors")
    ap.add_argument("--dry-run", action="store_true", help="print the batch plan without loading the model")
    return ap


def load_batch(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Batch file must contain a JSON list of tracks: {path}")
    return data


def track_slug(track: dict, index: int, used: set[str]) -> str:
    """Derive a unique, filesystem-friendly output subfolder from a track."""
    name = track.get("assigned_filename") or track.get("original_title") or f"track_{index:02d}"
    slug = Path(str(name)).stem.split("_SONG")[0].strip().replace("/", "_") or f"track_{index:02d}"
    base, n = slug, 2
    while slug in used:
        slug = f"{base}_{n}"
        n += 1
    used.add(slug)
    return slug


def build_plan(tracks: list[dict], root: Path) -> list[tuple[int, dict, Path]]:
    used: set[str] = set()
    plan = []
    for i, track in enumerate(tracks, start=1):
        sub = track.get("out") or track_slug(track, i, used)
        plan.append((i, track, root / sub))
    return plan


def main() -> None:
    args = build_parser().parse_args()
    ode_steps, also_legacy_vae = solo.effective_settings(args)

    tracks = load_batch(args.batch)
    if args.limit is not None:
        tracks = tracks[: args.limit]

    root = Path(args.out)
    plan = build_plan(tracks, root)

    print(f"BATCH  : {args.batch} ({len(plan)} tracks)")
    print(f"OUTPUT : {root}")
    print(f"SEED   : {args.seed} | CFG:", args.cfg, "| CoT:", args.cot,
          "| QUALITY:", args.quality, "(ode_steps=" + str(ode_steps) + ")")

    if args.dry_run:
        for i, track, out in plan:
            print(f"  [{i:02d}] {track.get('original_title', f'track {i}')} -> {out}")
        return

    t0 = time.perf_counter()
    pipe = solo.load_pipeline(args.model, args.vae, ode_steps)
    print(f"\n[loaded pipeline in {time.perf_counter() - t0:.1f}s]", flush=True)

    results = []
    for i, track, out in plan:
        title = track.get("original_title", f"track {i}")
        seed = track.get("seed", args.seed)
        cfg = track.get("cfg", args.cfg)
        cot = track.get("cot", args.cot)

        print("\n" + "=" * 72)
        print(f"[{i}/{len(plan)}] {title} -> {out}")

        if (out / "song.flac").exists() and not args.overwrite:
            print("song.flac exists, skipping (use --overwrite to regenerate)")
            results.append({"index": i, "title": title, "out": str(out), "status": "skipped"})
            continue

        missing = [k for k in ("styles", "lyrics") if not track.get(k)]
        if missing:
            print(f"ERROR: track missing {', '.join(missing)}")
            results.append({"index": i, "title": title, "out": str(out),
                            "status": "error", "error": f"missing {', '.join(missing)}"})
            if args.fail_fast:
                break
            continue

        solo.describe(title, track["styles"], track["lyrics"], seed, cfg, cot, args.quality, ode_steps)
        try:
            meta = solo.generate_song(
                pipe, title=title, style=track["styles"], lyrics=track["lyrics"], out=out,
                seed=seed, cfg=cfg, cot=cot, quality=args.quality, ode_steps=ode_steps,
                vae_legacy=args.vae_legacy, also_legacy_vae=also_legacy_vae,
            )
            results.append({"index": i, "title": title, "out": str(out), "status": "ok",
                            "generation_seconds": meta["generation_seconds"],
                            "peak_vram_gib": meta["peak_vram_gib"]})
        except Exception as exc:  # keep the batch going unless --fail-fast
            print(f"ERROR: {type(exc).__name__}: {exc}")
            results.append({"index": i, "title": title, "out": str(out),
                            "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                break

    pipe.close()

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    total = time.perf_counter() - t0
    summary = {
        "batch_file": str(args.batch),
        "out": str(root),
        "seed": args.seed, "cfg": args.cfg, "cot": args.cot,
        "quality": args.quality, "ode_steps": ode_steps,
        "model": args.model, "vae": args.vae, "vae_legacy": args.vae_legacy,
        "tracks_total": len(plan), "ok": ok, "skipped": skipped, "failed": failed,
        "wall_seconds": total, "results": results,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "batch_meta.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 72)
    print(f"BATCH DONE in {total:.1f}s: {ok} ok, {skipped} skipped, {failed} failed"
          f" -> {root / 'batch_meta.json'}")


if __name__ == "__main__":
    main()
