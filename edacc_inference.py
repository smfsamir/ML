"""
edacc_eval.py

Builds a filtered version of the EdACC dataset (edinburghcstr/edacc) and
evaluates a set of ASR models against it, writing per-sample results to
RESULT_CSV (appended incrementally so the run can be resumed after a crash
or OOM).

Dataset filtering pipeline (get_preprocessed_dataset):

  1. initial_filter   – drop samples with very short transcripts (<= 3 words),
                         metalinguistic tags (e.g. <laugh>, <no-speech>), or
                         audio shorter than 1 second.

  2. interquartile_filter – drop samples whose audio duration falls outside
                         the [Q1, Q3] range for their accent.

  3. accent_median_filter – drop entire accents whose median duration is
                         >= 10 seconds (intended to remove accents like
                         Chinese and Bulgarian that skew long).

Model loading is lazy: each ASR module is only imported (and its weights
loaded into memory) right before that model's evaluation loop runs, and
torch/process-level state is cleared again immediately after, so models
that are commented out — or simply not reached yet — never get loaded.

This is a non-streaming, in-memory pipeline — run it on a machine with
enough RAM/disk for the full dataset.

Usage:
    python edacc_eval.py
    python edacc_eval.py --split train
    python edacc_eval.py --median_threshold 10

Requirements:
    pip install datasets pandas tqdm torch
"""

from dotenv import dotenv_values
import traceback
import argparse
import gc
import json
import os
import sys
import unicodedata
import re

import torch
import pandas as pd
from collections import namedtuple
from fractions import Fraction
from tqdm import tqdm

from scripts.eval.metrics import cer, wer

CONFIG = dotenv_values(".env")  # config = {"SCRATCH_DIR"}
MODEL_WEIGHT_DIR = os.path.join(CONFIG["SCRATCH_DIR"], "model_weights")
DATASET_DIR = os.path.join(CONFIG["SCRATCH_DIR"], "datasets")

with open("spelling_variants.json") as f:
    SPELLING_VARIANTS = json.load(f)

# from https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/normalizer.py
def normalize_english_numbers(text: str):
    """Convert any spelled-out numbers into arabic numbers, remove any commas, keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.,
    spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`, spell out `one` and `ones`,
    interpret successive single-digit numbers as nominal: `one oh one` -> `101`"""

    ZEROS = {"o", "oh", "zero"}
    ONES = {name: i for i, name in enumerate(["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",],start=1,)}  # fmt: skip
    ONES_SUFFIXED = {**{"sixes" if name == "six" else name + "s": (value, "s")for name, value in ONES.items()}, **{"zeroth": (0, "th"),"first": (1, "st"),"second": (2, "nd"),"third": (3, "rd"),"fifth": (5, "th"),"twelfth": (12, "th"), **{name + ("h" if name.endswith("t") else "th"): (value, "th")for name, value in ONES.items()if value > 3 and value != 5 and value != 12},},}  # fmt: skip
    TENS = {"twenty": 20,"thirty": 30,"forty": 40,"fifty": 50,"sixty": 60,"seventy": 70,"eighty": 80,"ninety": 90}  # fmt: skip
    TENS_SUFFIXED = {**{name.replace("y", "ies"): (value, "s") for name, value in TENS.items()}, **{name.replace("y", "ieth"): (value, "th") for name, value in TENS.items()}}  # fmt: skip
    DECIMALS = {*ZEROS, *ONES, *TENS}
    MULTIPLIERS = {"hundred": 100,"thousand": 1_000,"million": 1_000_000,"billion": 1_000_000_000,"trillion": 1_000_000_000_000,"quadrillion": 1_000_000_000_000_000,"quintillion": 1_000_000_000_000_000_000,"sextillion": 1_000_000_000_000_000_000_000,"septillion": 1_000_000_000_000_000_000_000_000,"octillion": 1_000_000_000_000_000_000_000_000_000,"nonillion": 1_000_000_000_000_000_000_000_000_000_000,"decillion": 1_000_000_000_000_000_000_000_000_000_000_000,}  # fmt: skip
    MULTIPLIERS_SUFFIXED = {**{name + "s": (value, "s") for name, value in MULTIPLIERS.items()}, **{name + "th": (value, "th") for name, value in MULTIPLIERS.items()}}  # fmt: skip

    PRECEDING_PREFIXES = {"minus": "-","negative": "-","plus": "+","positive": "+"}  # fmt: skip
    FOLLOWING_PREFIXES = {"pound": "£","pounds": "£","euro": "€","euros": "€","dollar": "$","dollars": "$","cent": "¢","cents": "¢"}  # fmt: skip
    PREFIXES = set(list(PRECEDING_PREFIXES.values()) + list(FOLLOWING_PREFIXES.values()))  # fmt: skip
    SUFFIXES = {"per": {"cent": "%"},"percent": "%"}  # fmt: skip
    SPECIALS = {"and", "double", "triple", "point"}
    WORDS = {key for mapping in [ZEROS,ONES,ONES_SUFFIXED,TENS,TENS_SUFFIXED,MULTIPLIERS,MULTIPLIERS_SUFFIXED,PRECEDING_PREFIXES,FOLLOWING_PREFIXES,SUFFIXES,SPECIALS, ] for key in mapping}  # fmt: skip
    with open("spelling_variants.json") as f:
        SPELLING_VARIANTS = json.load(f)

    # replace "<number> and a half" with "<number> point five"
    results = []
    segments = re.split(r"\band\s+a\s+half\b", text)
    for i, segment in enumerate(segments):
        if len(segment.strip()) == 0:
            continue
        if i == len(segments) - 1:
            results.append(segment)
        else:
            results.append(segment)
            last_word = segment.rsplit(maxsplit=2)[-1]
            if last_word in DECIMALS or last_word in MULTIPLIERS:
                results.append("point five")
            else:
                results.append("and a half")
    text = " ".join(results)

    # put a space at number/letter boundary
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)

    # but remove spaces which could be a suffix
    text = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", text)

    def process_words(words: list[str]):
        prefix: "str | None" = None
        value: "str | int | None" = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for i, current in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in PREFIXES
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in WORDS:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in ZEROS:
                value = str(value or "") + "0"
            elif current in ONES:
                ones = ONES[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in ONES:
                    if (
                        prev in TENS and ones < 10
                    ):  # replace the last zero with the digit
                        value = value[:-1] + str(ones)  # type: ignore
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in ONES_SUFFIXED:
                # ordinal or cardinal; yield the number right away
                ones, suffix = ONES_SUFFIXED[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in ONES:
                    if prev in TENS and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)  # type: ignore
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in TENS:
                tens = TENS[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in TENS_SUFFIXED:
                # ordinal or cardinal; yield the number right away
                tens, suffix = TENS_SUFFIXED[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in MULTIPLIERS:
                multiplier = MULTIPLIERS[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)  # type: ignore
                    p = f * multiplier if f is not None else None
                    if p is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in MULTIPLIERS_SUFFIXED:
                multiplier, suffix = MULTIPLIERS_SUFFIXED[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if p is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in PRECEDING_PREFIXES:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in WORDS or next_is_numeric:
                    prefix = PRECEDING_PREFIXES[current]
                else:
                    yield output(current)
            elif current in FOLLOWING_PREFIXES:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = FOLLOWING_PREFIXES[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in SUFFIXES:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = SUFFIXES[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in SPECIALS:
                if next not in WORDS and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in MULTIPLIERS:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in ONES or next in ZEROS:
                        repeats = 2 if current == "double" else 3
                        ones = ONES.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in DECIMALS or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    text = " ".join(word for word in process_words(text.split()) if word is not None)

    # normalize currencies: "$2 and ¢7" -> "$2.07"
    def combine_cents(m):
        try:
            currency = m.group(1)
            integer = m.group(2)
            cents = int(m.group(3))
            return f"{currency}{integer}.{cents:02d}"
        except ValueError:
            return m.string

    def extract_cents(m):
        try:
            return f"¢{int(m.group(1))}"
        except ValueError:
            return m.string

    text = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, text)
    text = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, text)

    # write "one(s)" instead of "1(s)", just for the readability
    text = re.sub(r"\b1(s?)\b", r"one\1", text)

    return text

def normalize_english(text: str):
    text = text.lower()

    text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)  # remove words between brackets
    text = re.sub(r"\(([^)]+?)\)", "", text)  # remove words between parenthesis
    text = re.sub(r"\b(hmm|mm|mhm|mmm|uh|um)\b", "", text)  # remove uh and mms

    # standardize when there's a space before an apostrophe
    text = re.sub(r"\s+'", "'", text)

    # standardize contractions
    CONTRACTIONS = {r"\bwon't\b": "will not",r"\bcan't\b": "can not",r"\blet's\b": "let us",r"\bain't\b": "aint",r"\by'all\b": "you all",r"\bwanna\b": "want to",r"\bgotta\b": "got to",r"\bgonna\b": "going to",r"\bi'ma\b": "i am going to",r"\bimma\b": "i am going to",r"\bwoulda\b": "would have",r"\bcoulda\b": "could have",r"\bshoulda\b": "should have",r"\bma'am\b": "madam",r"\bmr\b": "mister ",r"\bmrs\b": "missus ",r"\bst\b": "saint ",r"\bdr\b": "doctor ",r"\bprof\b": "professor ",r"\bcapt\b": "captain ",r"\bgov\b": "governor ",r"\bald\b": "alderman ",r"\bgen\b": "general ",r"\bsen\b": "senator ",r"\brep\b": "representative ",r"\bpres\b": "president ",r"\brev\b": "reverend ",r"\bhon\b": "honorable ",r"\basst\b": "assistant ",r"\bassoc\b": "associate ",r"\blt\b": "lieutenant ",r"\bcol\b": "colonel ",r"\bjr\b": "junior ",r"\bsr\b": "senior ",r"\besq\b": "esquire ",r"'d been\b": " had been",r"'s been\b": " has been",r"'d gone\b": " had gone",r"'s gone\b": " has gone",r"'d done\b": " had done",r"'s got\b": " has got",r"n't\b": " not",r"'re\b": " are",r"'s\b": " is",r"'d\b": " would",r"'ll\b": " will",r"'t\b": " not",r"'ve\b": " have",r"'m\b": " am"}  # fmt: skip
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # remove commas between digits and periods not followed by numbers
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\.([^0-9]|$)", r" \1", text)

    # normalize unicode, keep some symbols for numerics, otherwise replace markers, symbols, and punctuations with a space (MSP) and drop diacritics (Mn + manual mappings)
    ADDITIONAL_DIACRITICS = {"œ": "oe","Œ": "OE","ø": "o","Ø": "O","æ": "ae","Æ": "AE","ß": "ss","ẞ": "SS","đ": "d","Đ": "D","ð": "d","Ð": "D","þ": "th","Þ": "th","ł": "l","Ł": "L"}  # fmt: skip
    text = "".join(
        (
            c
            if c in ".%$¢€£"
            else ADDITIONAL_DIACRITICS.get(
                c,
                (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                ),
            )
        )
        for c in unicodedata.normalize("NFKD", text)
    )

    text = normalize_english_numbers(text)
    text = " ".join(SPELLING_VARIANTS.get(word, word) for word in text.split())

    # now remove prefix/suffix symbols that are not preceded/followed by numbers
    text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
    text = re.sub(r"([^0-9])%", r"\1 ", text)

    # replace any successive whitespace characters with a space
    text = re.sub(r"\s+", " ", text)

    return text


RESULT_CSV = os.path.join("..", "..", ".data", "scores_asr.csv")

RESULT_COLUMNS = [
    "dataset", "model", "release_date", "sample_id", "native_language",
    "groundtruth", "prediction", "wer", "cer",
]


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Filter and evaluate ASR models on EdACC")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to load (default: test)")
    parser.add_argument("--median_threshold", type=float, default=10.0,
                        help="Drop accents with median duration >= this many seconds (default: 10.0)")
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of processes for .filter() (default: None, single process)")
    parser.add_argument("--cache_dir", type=str, default=DATASET_DIR,
                        help="Directory to save/load the filtered dataset (default: ../../.data/edacc_preprocessed)")
    parser.add_argument("--force_reprocess", action="store_true",
                        help="Ignore any cached preprocessed dataset and redo filtering from scratch")
    return parser.parse_args()


def initial_filter(row):
    """
    Returns True to KEEP the sample.

    Drops samples that:
      - have a transcript of 3 words or fewer
      - contain metalinguistic tags like <laugh>, <no-speech>
      - have audio shorter than 1 second
    """
    transcript = row["text"]
    audio = row["audio"]

    if len(transcript.split(" ")) <= 3:
        return False

    if "<" in transcript and ">" in transcript:
        return False

    audio_array, sampling_rate = audio["array"], audio["sampling_rate"]
    if len(audio_array) < sampling_rate:
        return False

    return True


def get_preprocessed_dataset(args):
    """
    Load edinburghcstr/edacc and apply the three-stage filtering pipeline:
    initial_filter -> interquartile_filter -> accent_median_filter.

    If a previously-saved, filtered dataset exists at args.cache_dir (and
    --force_reprocess wasn't passed), it's loaded directly via
    load_from_disk instead of redoing the filtering pass. Otherwise the
    filtering is run and the result is saved to args.cache_dir for next time.

    Returns the filtered HF Dataset.
    """
    from datasets import load_dataset, load_from_disk

    if not args.force_reprocess and os.path.exists(args.cache_dir):
        print(f"Found cached preprocessed dataset at '{args.cache_dir}', loading that instead of refiltering …")
        ds = load_from_disk(args.cache_dir)
        print(f"Loaded cached dataset: {len(ds)} samples, "
              f"{len(set(ds['accent']))} accents: {sorted(set(ds['accent']))}")
        return ds

    print(f"Loading edinburghcstr/edacc ({args.split} split) — this may take a while …")
    ds = load_dataset("edinburghcstr/edacc", split=args.split, cache_dir=DATASET_DIR)
    print(f"Initial size: {len(ds)}")

    # --- 1. initial filter -----------------------------------------------------
    print("\nApplying initial filter (short transcripts, metalinguistic tags, short audio) …")
    ds = ds.filter(initial_filter, num_proc=args.num_proc)
    print(f"After initial filter: {len(ds)}")

    # --- compute durations once, used by filters 2 and 3 ----------------------
    print("\nComputing durations …")
    durations = [len(row["audio"]["array"]) / row["audio"]["sampling_rate"] for row in ds]
    accents = ds["accent"]
    df = pd.DataFrame({"accent": accents, "duration_s": durations})

    # --- 2. interquartile filter (per accent) ----------------------------------
    print("\nApplying interquartile filter (per-accent Q1–Q3 duration range) …")
    quartiles = df.groupby("accent")["duration_s"].quantile([0.25, 0.75]).unstack()
    quartiles.columns = ["q1", "q3"]
    bounds = {accent: (row["q1"], row["q3"]) for accent, row in quartiles.iterrows()}

    def interquartile_filter(row):
        accent = row["accent"]
        q1, q3 = bounds[accent]
        duration_s = len(row["audio"]["array"]) / row["audio"]["sampling_rate"]
        return q1 <= duration_s <= q3 and duration_s <= 30

    ds = ds.filter(interquartile_filter, num_proc=args.num_proc)
    print(f"After interquartile filter: {len(ds)}")

    # --- 3. accent median duration filter ---------------------------------------
    print(f"\nApplying accent median duration filter (drop accents with median >= {args.median_threshold}s) …")
    durations2 = [len(row["audio"]["array"]) / row["audio"]["sampling_rate"] for row in ds]
    accents2 = ds["accent"]
    df2 = pd.DataFrame({"accent": accents2, "duration_s": durations2})
    medians = df2.groupby("accent")["duration_s"].median()

    dropped_accents = medians[medians >= args.median_threshold].index.tolist()
    kept_accents = set(medians[medians < args.median_threshold].index)

    if dropped_accents:
        print(f"  Dropping accents (median >= {args.median_threshold}s): {dropped_accents}")
    else:
        print("  No accents exceeded the median duration threshold.")

    def accent_median_filter(row):
        return row["accent"] in kept_accents

    ds = ds.filter(accent_median_filter, num_proc=args.num_proc)
    print(f"After accent median filter: {len(ds)}")

    print(f"\nFinal accents ({len(set(ds['accent']))}): {sorted(set(ds['accent']))}")
    print(f"Final size: {len(ds)}")

    print(f"\nSaving preprocessed dataset to '{args.cache_dir}' for reuse next time …")
    os.makedirs(os.path.dirname(args.cache_dir) or ".", exist_ok=True)
    ds.save_to_disk(args.cache_dir)

    return ds


# ---------------------------------------------------------------------------
# Lazy model registry.
#
# Each entry is (name, release_date, load_fn). load_fn is a zero-argument
# callable that performs the module import (which is what actually pulls
# weights into memory) and returns the transcribe_fn for that model. The
# import is deferred until the model is actually about to be evaluated, so
# commented-out models — or models not yet reached in the loop — never get
# loaded.
# ---------------------------------------------------------------------------

def _load_whisper(model_size):
    from scripts.asr.whisper import whisper_transcribe_from_array, whisper_output_to_text
    return lambda arr: whisper_output_to_text(
        whisper_transcribe_from_array(arr, model=model_size, language='en')
    )

def _load_mms():
    from scripts.asr.mms import mms_transcribe_from_array
    return mms_transcribe_from_array

def _load_owsm():
    from scripts.asr.owsm import owsm_transcribe_from_array
    return owsm_transcribe_from_array

def _load_qwen2audio():
    from scripts.asr.qwen2audio import qwen_transcribe_from_array
    return qwen_transcribe_from_array

def _load_qwen3asr():
    from scripts.asr.qwen3asr import qwen_asr_transcribe_from_array
    return qwen_asr_transcribe_from_array

def _load_canary():
    from scripts.asr.canary import canary_transcribe_from_array
    return canary_transcribe_from_array

def _load_deepspeech():
    from scripts.asr.deepspeech import deepspeech_transcribe_from_array
    return deepspeech_transcribe_from_array

def _load_omni():
    from scripts.asr.omni import omni_transcribe_from_array
    return omni_transcribe_from_array

def _load_wavlm():
    from scripts.asr.wavlm import wavlm_transcribe_from_array
    return wavlm_transcribe_from_array

def _load_seamlessm4t():
    from scripts.asr.seamlessm4t import seamlessm4t_transcribe_from_array
    return seamlessm4t_transcribe_from_array

def _load_google():
    from scripts.asr.google_speech import google_transcribe_from_array
    return lambda arr: google_transcribe_from_array(arr).results[0].alternatives[0].transcript


def get_models():
    Model = namedtuple('Model', ['name', 'release_date', 'load_fn'])

    models = [
        Model('Whisper Large v1', '2022-09', lambda: _load_whisper('large')),
        Model('Whisper Large v2', '2022-12', lambda: _load_whisper('large-v2')),
        Model('MMS 1B All',       '2023-05', _load_mms),
        Model('Whisper Large v3', '2023-11', lambda: _load_whisper('large-v3')),
        # Model('OWSM 1.0 272M',    '2023-09', _load_owsm),
        # # Model('OWSM 2.0 739M',    '2023-10', _load_owsm),
        # # Model('OWSM 3.1 1B',      '2024-01', _load_owsm),
        Model('OWSM 4.0 1B',      '2025-05', _load_owsm),
        # Model('Qwen2 Audio 7B',   '2024-08', _load_qwen2audio),
        Model('Qwen3 ASR 1.7B',   '2025-06', _load_qwen3asr),
        # Model('Canary Qwen 2.5B', '2025-07', _load_canary),
        Model('Deepspeech',       '2021-10', _load_deepspeech),
        # Model('Omnilingual 7B',   '2025-11', _load_omni),
        # Model('Google Chirp 3',   '2025-10', _load_google),
        # Model('WavLM Libri-100h', '2021-12', _load_wavlm),
        Model('SeamlessM4T',      '2023-12', _load_seamlessm4t),
    ]
    return models


def load_completed_keys(result_csv: str) -> set:
    """
    Read the results CSV (if it exists) and return a set of
    (model, sample_id) pairs that have already been evaluated, so a
    crashed/interrupted run can resume without redoing work.
    """
    if not os.path.exists(result_csv):
        return set()

    try:
        existing = pd.read_csv(result_csv, names=RESULT_COLUMNS, header=None)
    except Exception as e:
        print(f"Warning: could not read existing results CSV ({e}); starting fresh.")
        return set()

    return set(zip(existing["model"], existing["sample_id"]))


def append_result_row(result_csv: str, row: dict):
    new_row = pd.DataFrame({k: [v] for k, v in row.items()}, columns=RESULT_COLUMNS)
    write_header = not os.path.exists(result_csv)
    new_row.to_csv(result_csv, mode="a", index=False, header=write_header)


def run_model_on_dataset(model_name, model_date, transcribe_fn, dataset, completed_keys):
    """
    Run a single (already-loaded) model over the full dataset, writing each
    result to RESULT_CSV as soon as it's computed. Samples already present
    in completed_keys (from a previous run) are skipped.

    A tqdm progress bar tracks progress over the dataset. If an
    individual sample raises (e.g. CUDA OOM on a long clip), the error
    is logged, the cache is cleared, and the loop continues so one bad
    sample doesn't kill the whole run.
    """
    os.makedirs(os.path.dirname(RESULT_CSV) or ".", exist_ok=True)

    n_skipped = 0
    n_done = 0
    n_errors = 0

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc=model_name, unit="sample")
    for sample_id, sample in pbar:
        if (model_name, sample_id) in completed_keys:
            n_skipped += 1
            continue

        gt_transcript = sample["text"]
        normalized_groundtruth = normalize_english(gt_transcript)
        if not normalized_groundtruth:
            # skip <laugh>, <cough>, etc. — shouldn't occur post-filtering,
            # but keep the guard for safety
            continue

        accent = sample["accent"]
        audio = sample["audio"]["array"]

        try:
            clear_cache()
            predicted_transcription = transcribe_fn(audio)
            normalized_transcription = normalize_english(predicted_transcription)

            row = dict(
                dataset="edacc",
                model=model_name,
                release_date=model_date,
                sample_id=sample_id,
                native_language=accent,
                groundtruth=gt_transcript,
                prediction=predicted_transcription,
                wer=wer(normalized_transcription, normalized_groundtruth),
                cer=cer(normalized_transcription, normalized_groundtruth),
            )
            append_result_row(RESULT_CSV, row)
            completed_keys.add((model_name, sample_id))
            n_done += 1

        except Exception as e:
            n_errors += 1
            tqdm.write(f"  [{model_name}] sample {sample_id} failed: {type(e).__name__}: {e}")
            clear_cache()
            continue

        pbar.set_postfix(done=n_done, skipped=n_skipped, errors=n_errors)

    print(f"  {model_name}: {n_done} done, {n_skipped} skipped (resumed), {n_errors} errors")


def main():
    args = parse_args()
    dataset = get_preprocessed_dataset(args)
    models = get_models()

    completed_keys = load_completed_keys(RESULT_CSV)
    if completed_keys:
        print(f"\nFound {len(completed_keys)} previously-completed (model, sample) results in {RESULT_CSV}")
        print("Resuming — already-completed samples will be skipped.\n")

    for model in models:
        model_name = model.name
        print(f"\n{'=' * 60}")
        print(f"Running model: {model_name} (released {model.release_date})")
        print('=' * 60)

        transcribe_fn = None
        try:
            print(f"  Loading {model_name} …")
            transcribe_fn = model.load_fn()
            print(f"  {model_name} loaded.")

            run_model_on_dataset(model_name, model.release_date, transcribe_fn, dataset, completed_keys)

        except Exception as e:
            print(f"  Model {model_name} crashed: {type(e).__name__}: {e}")
            print("  Skipping to next model. Already-saved results for this model are preserved.")
            # print traceback for debugging, but keep going so one bad model doesn't kill the whole run
            traceback.print_exc()


        finally:
            # Drop the reference to the loaded model/transcribe_fn and force
            # a cleanup pass so the next model's weights have room, whether
            # this model succeeded, partially completed, or crashed.
            del transcribe_fn
            clear_cache()
            print(f"  {model_name} unloaded, cache cleared.")


if __name__ == "__main__":
    main()