#!/usr/bin/env python3

import sys
import os
from tempfile import NamedTemporaryFile

import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from dotenv import dotenv_values

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from core.audio import audio_record_to_file

TARGET_SR = 16000

_model_size = "small"  # other options: large-v1, large-v2, large-v3
_pipe = None


def get_model(model_size):
    """Load (and cache) a HF ASR pipeline for the requested Whisper size."""
    CONFIG = dotenv_values(".env")  # config = {"SCRATCH_DIR"}
    global _pipe, _model_size

    if _pipe is not None and model_size == _model_size:
        return _pipe

    _model_size = model_size
    model_id = f"openai/whisper-{model_size}"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        cache_dir=CONFIG.get("SCRATCH_DIR"),
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=CONFIG.get("SCRATCH_DIR"))
    processor.tokenizer.clean_up_tokenization_spaces = False


    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )
    return _pipe


def _prepare_array(wav_array, sampling_rate):
    """Ensure a mono float32 array at 16 kHz."""
    wav_array = np.asarray(wav_array, dtype=np.float32)
    if wav_array.ndim > 1:
        wav_array = wav_array.mean(axis=1)  # collapse to mono if stereo
    if sampling_rate != TARGET_SR:
        import librosa  # lazy import; only needed for non-16kHz input
        wav_array = librosa.resample(
            wav_array, orig_sr=sampling_rate, target_sr=TARGET_SR
        )
    return wav_array


def _run(audio, model, language, timestamped):
    pipe = get_model(model)
    generate_kwargs = {"task": "transcribe"}
    if language is not None:
        generate_kwargs["language"] = language
    return pipe(
        audio,
        return_timestamps=("word" if timestamped else False),
        generate_kwargs=generate_kwargs,
    )


def whisper_transcribe(input_path, model="small", language="en"):
    # The pipeline decodes the file via ffmpeg and resamples to 16kHz itself.
    return _run(input_path, model, language, timestamped=False)


def whisper_transcribe_timestamped(input_path, model="small", language="en"):
    return _run(input_path, model, language, timestamped=True)


def whisper_transcribe_from_array(wav_array, model="small", language="en", sampling_rate=TARGET_SR):
    return _run(_prepare_array(wav_array, sampling_rate), model, language, timestamped=False)


def whisper_transcribe_timestamped_from_array(wav_array, model="small", language="en", sampling_rate=TARGET_SR):
    return _run(_prepare_array(wav_array, sampling_rate), model, language, timestamped=True)


def whisper_transcribe_from_mic(model="small", language="en"):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return whisper_transcribe(f.name, model=model, language=language)


def whisper_transcribe_timestamped_from_mic(model="small", language="en"):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_record_to_file(f.name)
        return whisper_transcribe_timestamped(f.name, model=model, language=language)


def whisper_output_to_text(output, timestamped=False):
    # HF pipeline returns: {"text": ..., "chunks": [{"text", "timestamp"}, ...]}
    if isinstance(output, dict):
        if timestamped and output.get("chunks"):
            return " ".join(chunk["text"].strip() for chunk in output["chunks"])
        return output.get("text", "").strip()
    if hasattr(output, "text"):
        return output.text
    return str(output)


def display_whisper_result(output, timestamped=False):
    if timestamped and isinstance(output, dict) and output.get("chunks"):
        for chunk in output["chunks"]:
            ts = chunk.get("timestamp", (None, None))
            start, end = ts if ts else (None, None)
            if start is not None and end is not None:
                print("[%.2fs -> %.2fs] %s" % (start, end, chunk["text"].strip()))
            else:
                print(chunk["text"].strip())
    else:
        print(whisper_output_to_text(output, timestamped))


def main(args):
    timestamped = len(args) > 1 and args[1] == "--timestamped"
    try:
        if args[0] == "mic":
            if timestamped:
                result = whisper_transcribe_timestamped_from_mic()
            else:
                result = whisper_transcribe_from_mic()
        else:
            input_path = args[0]
            if timestamped:
                result = whisper_transcribe_timestamped(input_path)
            else:
                result = whisper_transcribe(input_path)
        display_whisper_result(result, timestamped)
    except Exception as e:
        print(e)
        print("Usage: python ./scripts/asr/whisper.py mic [--timestamped]")
        print("Usage: python ./scripts/asr/whisper.py <input_wav_path> [--timestamped]")


if __name__ == "__main__":
    main(sys.argv[1:])