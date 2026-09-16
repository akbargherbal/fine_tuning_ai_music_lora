#!/usr/bin/env python3
"""YuE2-3B base-model bias probe (Arabic/Khaliji/Nasheed vs. Western idioms).

Loads the base YuE2-3B model ONCE (no LoRA) and runs a fixed 10-case probe
matrix through it, to answer two questions before any further LoRA work:

  1. Is the base model biased toward Western melodic/rhythmic idioms even
     when the style prompt explicitly asks for Arabic/Khaliji/Nasheed music?
  2. Is the "drums driving the vocal" behavior seen in the live corpus
     caption (`stately groove` / `tight rhythm section`) a property of that
     specific phrasing, or does it show up under ANY Western-rock framing?

All 10 cases share the same lyrics (Track 07's poem, verbatim from
`workspace_manifest.json`) and the same seed, so the only thing that varies
between renders is the style prompt. Outputs are written under randomized
4-letter blind codes with the real mapping kept in a separate
`blind_key.json`, so you can listen and take notes before knowing which
render is which.

Usage
-----
Review the 10 prompts without spending any GPU time:
    python yue2_bias_probe.py --dry-run

Run the full matrix (fast preset, ~10 renders):
    python yue2_bias_probe.py --out outputs/bias_probe

Run only some cases (see --list-cases for ids):
    python yue2_bias_probe.py --cases control,khaliji,nasheed,tilawah --out outputs/bias_probe

Higher quality:
    python yue2_bias_probe.py --quality high --out outputs/bias_probe_high
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

MODEL = "/content/models/YuE2-3B"
VAE = "/content/models/YuE2-Vae"

QUALITY_PRESETS = {
    "standard": {"ode_steps": 32},
    "high": {"ode_steps": 48},
    "max": {"ode_steps": 64},
}

# ---------------------------------------------------------------------------
# Shared lyrics: Track 07 ("يوم بدر الثاني وتتويج نصر العرب"), verbatim from
# workspace_manifest.json. Held constant across every vocal case so style
# prompt is the only variable.
# ---------------------------------------------------------------------------
FULL_LYRICS = """///***///
[Intro | single clean guitar | close-mic'd, plate reverb, short decay]
كَمْ نِيلَ تَحْتَ سَنَاهَا مِنْ سَنَا قَمَرٍ
وَتَحْتَ عَارِضِهَا مِنْ عَارِضٍ شَنِبِ...

[Verse 1 | noble triumphant vocals | driving rock guitars and rich orchestral brass]
كَمْ نِيلَ تَحْتَ سَنَاهَا مِنْ سَنَا قَمَرٍ
وَتَحْتَ عَارِضِهَا مِنْ عَارِضٍ شَنِبِ...
كَمْ كَانَ فِي قَطْعِ أَسْبَابِ الرِّقَابِ بِهَا
إِلَى الْمُخَدَّرَةِ الْعَذْرَاءِ مِنْ سَبَبِ...
كَمْ أَحْرَزَتْ قُضُبُ الْهِنْدِيِّ مُصْلَتَةً
تَهْتَزُّ مِنْ قُضُبٍ تَهْتَزُّ فِي كُثُبِ...
بِيضٌ إِذَا انْتُضِيَتْ مِنْ حُجْبِهَا رَجَعَتْ
أَحَقَّ بِالْبِيضِ أَتْرَاباً مِنَ الْحُجُبِ...

[Chorus | majestic soaring vocals | massive orchestral swell and soaring guitars — Ajam]
إِنْ كَانَ بَيْنَ صُرُوفِ الدَّهْرِ مِنْ رَحِمٍ
مَوْصُولَةٍ أَوْ ذِمَامٍ غَيْرِ مُنْقَضِبِ...
فَبَيْنَ أَيَّامِكَ اللَّاتِي نُصِرْتَ بِهَا
وَبَيْنَ أَيَّامِ بَدْرٍ أَقْرَبُ النَّسَبِ...

[Verse 2]
خَلِيفَةَ اللَّهِ جَازَى اللَّهُ سَعْيَكَ عَنْ
جُرْثُومَةِ الدِّينِ وَالْإِسْلَامِ وَالْحَسَبِ...
بَصُرْتَ بِالرَّاحَةِ الْكُبْرَى فَلَمْ تَرَهَا
تُنَالُ إِلَّا عَلَى جِسْرٍ مِنَ التَّعَبِ...
أَبْقَتْ بَنِي الْأَصْفَرِ الْمِمْرَاضِ كَاسْمِهِمُ
صُفْرَ الْوُجُوهِ وَجَلَّتْ أَوْجُهَ الْعَرَبِ...

[Chorus]
إِنْ كَانَ بَيْنَ صُرُوفِ الدَّهْرِ مِنْ رَحِمٍ
مَوْصُولَةٍ أَوْ ذِمَامٍ غَيْرِ مُنْقَضِبِ...
فَبَيْنَ أَيَّامِكَ اللَّاتِي نُصِرْتَ بِهَا
وَبَيْنَ أَيَّامِ بَدْرٍ أَقْرَبُ النَّسَبِ...

[Outro | deep male vocals | clean electric guitar]
أَبْقَتْ بَنِي الْأَصْفَرِ الْمِمْرَاضِ كَاسْمِهِمُ
صُفْرَ الْوُجُوهِ وَجَلَّتْ أَوْجُهَ الْعَرَبِ..."""

# Minimal cue for the two instrumental-only cases -- nothing to sing, but a
# structure tag keeps YuE2's planner from inventing an unrelated form.
INSTRUMENTAL_CUE = "[Instrumental | no vocals | full arrangement]"

# ---------------------------------------------------------------------------
# The 10 probe cases. Each is (id, label, style, lyrics, notes).
# `style` fields are written in the same genre/vocals/production/
# instrumentation/mood shape as the live workspace_manifest.json template,
# so results stay comparable to what you already have.
# ---------------------------------------------------------------------------
CASES = [
    dict(
        id="control",
        label="Live template, unmodified",
        notes="Exact Track 07 style string, verbatim, incl. Suno control tags. "
              "This is the same prompt that produced your existing `base` "
              "render (groove-driving-vocal) -- reproduced here for continuity "
              "inside the same batch run.",
        lyrics=FULL_LYRICS,
        style=(
            '[Is_MAX_MODE: MAX](MAX) [QUALITY: MAX](MAX) [REALISM: MAX](MAX)\n'
            '[START_ON: TRUE]\n'
            '[START_ON: "كَمْ نِيلَ تَحْتَ"]\n\n'
            'genre: "Symphonic cinematic orchestral ballad, hymn-like grand concert '
            'hall acoustics, heavy rock instrumentation, stately groove, 110 BPM."\n'
            'vocals: "deep male vocals, mixed-voice chest-head resonance blend on '
            'sustained notes, breath-supported melismatic runs, controlled vibrato, '
            'full-voiced commanding presence, precise Arabic diction, melismatic '
            'phrasing in Maqam Ajam with unhurried phrase-ending sustains."\n'
            'production: "Audiophile recording, punchy centered mix, forward vocals '
            'pulling instrumentation down on sustained phrases then band re-enters '
            'between lines, bright presence, clean transients, large dynamic range, '
            'natural breath room between phrases."\n'
            'instrumentation: "Distorted electric guitars, orchestral strings, '
            'weighted acoustic rock drums, tight rhythm section."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="reworded",
        label="Live template, groove-lock phrases reworded",
        notes="Same as control, but 'stately groove' and 'tight rhythm section' "
              "(the two phrases suspected of overriding `production`'s "
              "vocal-led instruction) are swapped for wording that keeps the "
              "epic/orchestral weight without the beat-lock idiom. If the "
              "vocal-drum relationship loosens here relative to `control`, "
              "it's a caption-wording artifact, not a model limitation.",
        lyrics=FULL_LYRICS,
        style=(
            '[Is_MAX_MODE: MAX](MAX) [QUALITY: MAX](MAX) [REALISM: MAX](MAX)\n'
            '[START_ON: TRUE]\n'
            '[START_ON: "كَمْ نِيلَ تَحْتَ"]\n\n'
            'genre: "Symphonic cinematic orchestral ballad, hymn-like grand concert '
            'hall acoustics, heavy rock instrumentation, unhurried majestic pulse '
            'that follows the vocal, 110 BPM."\n'
            'vocals: "deep male vocals, mixed-voice chest-head resonance blend on '
            'sustained notes, breath-supported melismatic runs, controlled vibrato, '
            'full-voiced commanding presence, precise Arabic diction, melismatic '
            'phrasing in Maqam Ajam with unhurried phrase-ending sustains."\n'
            'production: "Audiophile recording, punchy centered mix, forward vocals '
            'pulling instrumentation down on sustained phrases then band re-enters '
            'between lines, bright presence, clean transients, large dynamic range, '
            'natural breath room between phrases."\n'
            'instrumentation: "Distorted electric guitars, orchestral strings, '
            'weighted acoustic rock drums following the vocal line, loose supportive '
            'rhythm section."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="khaliji",
        label="Khaliji / Gulf, positive descriptor",
        notes="Direct probe of the Suno-era trick: naming Khaliji/Saudi/Kuwaiti "
              "as POSITIVE genre tokens (not excludes) with traditional "
              "instrumentation, to see how far off its Western default the "
              "base model can be pulled.",
        lyrics=FULL_LYRICS,
        style=(
            'genre: "Khaliji Gulf Arabic vocal ballad, Saudi Kuwaiti musical '
            'tradition, cinematic scale, unhurried majestic pulse, 110 BPM."\n'
            'vocals: "deep male vocals, breath-supported melismatic runs, '
            'controlled vibrato, full-voiced commanding presence, precise Arabic '
            'diction, melismatic phrasing in Maqam Ajam with unhurried '
            'phrase-ending sustains."\n'
            'production: "Audiophile recording, punchy centered mix, forward '
            'vocals pulling instrumentation down on sustained phrases then band '
            're-enters between lines, bright presence, natural breath room '
            'between phrases."\n'
            'instrumentation: "oud, qanun, riq and traditional Khaliji hand '
            'percussion, layered beneath light orchestral strings."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="nasheed",
        label="Islamic Nasheed, positive descriptor",
        notes="Probes whether 'Islamic Nasheed' as a genre token pulls the "
              "model toward vocal-forward, light-percussion devotional "
              "phrasing the way it reportedly did in Suno 3.5.",
        lyrics=FULL_LYRICS,
        style=(
            'genre: "Islamic Nasheed, devotional vocal ballad, vocal-forward, '
            'unhurried, 110 BPM."\n'
            'vocals: "deep male vocals, breath-supported melismatic runs, '
            'controlled vibrato, full-voiced commanding presence, precise Arabic '
            'diction, Nasheed-style melismatic phrasing in Maqam Ajam with '
            'unhurried phrase-ending sustains."\n'
            'production: "Audiophile recording, intimate centered mix, forward '
            'vocals leading at all times, light accompaniment yielding fully to '
            'the voice on sustained phrases, natural breath room between '
            'phrases."\n'
            'instrumentation: "daf hand drum only, sparse and yielding to the '
            'vocal, no melodic instruments."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="tilawah",
        label="Quranic Tilawah-style vocals, no instrumentation",
        notes="Tests the specific Suno behavior the human described: 'Quran' "
              "and 'Tilawah, only vocals' as tokens reportedly locked in "
              "accurate Arabic pronunciation and produced an unaccompanied "
              "vocal texture. This is a vocal-recitation STYLE probe using "
              "the poem's text -- not an actual Quranic recitation.",
        lyrics=FULL_LYRICS,
        style=(
            'genre: "Quranic Tilawah style, solo vocal recitation, no music, no '
            'instrumentation, unaccompanied."\n'
            'vocals: "deep male vocals, Tajweed-style articulation, melismatic '
            'Maqam recitation, unaccompanied solo voice, precise Arabic diction, '
            'unhurried phrase-ending sustains, natural breath pauses between '
            'verses."\n'
            'production: "Close-mic\'d, dry and intimate, minimal reverb, solo '
            'voice only, no instrumentation, no rhythm section."\n'
            'instrumentation: "none -- vocal only."\n'
            'mood: "Solemn, Reverent, Exalted"'
        ),
    ),
    dict(
        id="takht",
        label="Classical Arabic Takht ensemble",
        notes="Probes a specific historical register (Umm Kulthum-era Takht "
              "ensemble) rather than a modern Gulf/devotional one, to see if "
              "the bias (if any) is uniform across Arabic sub-genres or "
              "concentrated in one direction.",
        lyrics=FULL_LYRICS,
        style=(
            'genre: "Classical Arabic Takht ensemble, mid-20th-century Cairo '
            'orchestral maqam music, unhurried, 110 BPM."\n'
            'vocals: "deep male vocals, breath-supported melismatic runs, '
            'controlled vibrato, precise Arabic diction, classical melismatic '
            'phrasing in Maqam Ajam with extended phrase-ending sustains."\n'
            'production: "Warm vintage recording, forward vocals, instrumentation '
            'yielding on sustained phrases then re-entering between lines, '
            'natural room ambience."\n'
            'instrumentation: "oud, qanun, nay, violin section, riq -- classical '
            'Takht ensemble."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="minimal",
        label="Near-empty prompt (unconditioned prior)",
        notes="Minimal conditioning -- almost nothing to steer the model. "
              "Shows YuE2's default/prior direction when not told which "
              "culture or genre to aim for.",
        lyrics=FULL_LYRICS,
        style='genre: "Arabic vocal song."',
    ),
    dict(
        id="instrumental_khaliji",
        label="Instrumental only -- Khaliji ensemble",
        notes="No vocals, so pronunciation/melisma can't mask or explain away "
              "rhythmic/melodic choices. Pairs with `instrumental_western` "
              "below for a clean instrumental-only A/B.",
        lyrics=INSTRUMENTAL_CUE,
        style=(
            'genre: "Khaliji Gulf instrumental, cinematic scale, unhurried, '
            '110 BPM."\n'
            'vocals: "instrumental only, no vocals."\n'
            'production: "Audiophile recording, wide stereo image, natural '
            'dynamics."\n'
            'instrumentation: "oud, qanun, riq, traditional Khaliji hand '
            'percussion, light orchestral strings."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="instrumental_western",
        label="Instrumental only -- current Western-rock instrumentation",
        notes="Same idea as `instrumental_khaliji`, using the LIVE template's "
              "instrumentation fields verbatim (minus vocals), to isolate "
              "whether the rhythmic drive lives in the instrumentation "
              "description itself, independent of any vocal interaction.",
        lyrics=INSTRUMENTAL_CUE,
        style=(
            'genre: "Symphonic cinematic orchestral instrumental, hymn-like '
            'grand concert hall acoustics, heavy rock instrumentation, stately '
            'groove, 110 BPM."\n'
            'vocals: "instrumental only, no vocals."\n'
            'production: "Audiophile recording, punchy centered mix, bright '
            'presence, clean transients, large dynamic range."\n'
            'instrumentation: "Distorted electric guitars, orchestral strings, '
            'weighted acoustic rock drums, tight rhythm section."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
    dict(
        id="western_control",
        label="Plain Western-rock control, Arabic vocals",
        notes="Broader/plainer Western-rock framing than `control` -- no "
              "cinematic/hymn dressing, no 'stately groove' phrase at all -- "
              "to check whether ANY generic Western-rock genre tag reproduces "
              "the groove-driving-vocal behavior, or whether it's specific to "
              "the live template's exact wording.",
        lyrics=FULL_LYRICS,
        style=(
            'genre: "Anthemic Western rock ballad, driving rhythm section, '
            '110 BPM."\n'
            'vocals: "deep male vocals, breath-supported melismatic runs, '
            'controlled vibrato, precise Arabic diction, melismatic phrasing in '
            'Maqam Ajam with unhurried phrase-ending sustains."\n'
            'production: "Punchy centered mix, bright presence, clean '
            'transients."\n'
            'instrumentation: "Electric guitars, rock drums, bass guitar."\n'
            'mood: "Exalted, Glorious, Triumphant"'
        ),
    ),
]

CASES_BY_ID = {c["id"]: c for c in CASES}


def random_code(existing: set[str]) -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        if code not in existing:
            return code


def list_cases() -> None:
    print(f"{'id':<22} label")
    print("-" * 70)
    for c in CASES:
        print(f"{c['id']:<22} {c['label']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all",
                     help="comma-separated case ids to run, or 'all' (default)")
    ap.add_argument("--list-cases", action="store_true", help="print case ids/labels and exit")
    ap.add_argument("--seed", type=int, default=831001, help="seed held constant across all cases")
    ap.add_argument("--cfg", type=float, default=1.2)
    ap.add_argument("--cot", default="full", choices=["full", "melody", "off"])
    ap.add_argument("--quality", default="standard", choices=list(QUALITY_PRESETS))
    ap.add_argument("--ode-steps", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--vae", default=VAE)
    ap.add_argument("--out", default="outputs/bias_probe")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the case list and full prompts, load nothing, generate nothing")
    ap.add_argument("--no-blind", action="store_true",
                     help="name output folders by case id instead of a random blind code "
                          "(skips blind_key.json)")
    args = ap.parse_args()

    if args.list_cases:
        list_cases()
        return

    if args.cases == "all":
        selected = list(CASES)
    else:
        ids = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = [i for i in ids if i not in CASES_BY_ID]
        if unknown:
            raise SystemExit(f"Unknown case id(s): {unknown}. Use --list-cases to see valid ids.")
        selected = [CASES_BY_ID[i] for i in ids]

    preset = QUALITY_PRESETS[args.quality]
    ode_steps = args.ode_steps if args.ode_steps is not None else preset["ode_steps"]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[DRY RUN] {len(selected)} case(s) selected, no model loaded, nothing generated.\n")
        for c in selected:
            print("=" * 78)
            print(f"id: {c['id']}   label: {c['label']}")
            print(f"notes: {c['notes']}")
            print("-" * 78)
            print("STYLE:\n" + c["style"])
            print("-" * 78)
            print("LYRICS:\n" + c["lyrics"][:200] + ("..." if len(c["lyrics"]) > 200 else ""))
            print()
        return

    # Blind coding: real case id/label only recorded in blind_key.json,
    # not in the folder name, so listening notes are taken unbiased.
    blind_key: dict[str, dict] = {}
    used_codes: set[str] = set()

    print(f"Running {len(selected)} case(s) | seed={args.seed} cfg={args.cfg} "
          f"cot={args.cot} quality={args.quality} (ode_steps={ode_steps})")
    sys.stdout.flush()

    import torch
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig

    t0 = time.perf_counter()
    config = GenerationConfig(ode_steps=ode_steps)
    pipe = YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda", generation_config=config)
    print(f"[loaded pipeline in {time.perf_counter() - t0:.1f}s]", flush=True)

    for i, case in enumerate(selected, 1):
        if args.no_blind:
            folder_name = case["id"]
        else:
            folder_name = random_code(used_codes)
            used_codes.add(folder_name)
            blind_key[folder_name] = {"id": case["id"], "label": case["label"], "notes": case["notes"]}

        out = out_root / folder_name
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i}/{len(selected)}] -> {folder_name}" + ("" if args.no_blind else f"  (blind code for '{case['id']}')"))
        print("STYLE:\n" + case["style"])
        sys.stdout.flush()

        t1 = time.perf_counter()
        song = pipe(style=case["style"], lyrics=case["lyrics"], cot=args.cot,
                    seed=args.seed, cfg_scale=args.cfg)
        gen = time.perf_counter() - t1

        song.save(str(out / "song.flac"))
        song.save_artifacts(str(out / "artifacts"))

        meta = {
            "case_id": case["id"] if args.no_blind else "REDACTED (see blind_key.json)",
            "style": case["style"],
            "lyrics": case["lyrics"],
            "seed": args.seed, "cfg_scale": args.cfg, "cot": args.cot,
            "quality": args.quality, "ode_steps": ode_steps,
            "generation_seconds": gen, "timing": song.timing,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        }
        (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"done in {gen:.1f}s -> {out / 'song.flac'}")

    if not args.no_blind:
        (out_root / "blind_key.json").write_text(json.dumps(blind_key, ensure_ascii=False, indent=2))
        print(f"\nBlind key written to {out_root / 'blind_key.json'} -- "
              f"don't open it until you've listened to all {len(selected)} renders.")

    print(f"\nDONE. {len(selected)} render(s) in {out_root}/")
    pipe.close()


if __name__ == "__main__":
    main()
