import io
import os
import tempfile
import traceback
import warnings

import librosa
import librosa.effects
import librosa.feature
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_runtime import (
    ARTIFACT_PATH,
    build_model_vector,
    load_model_artifact,
    severity_from_probability,
    train_and_save_model,
)

warnings.filterwarnings("ignore")

app = FastAPI(title="Parkinson's Voice Analysis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ARTIFACT = load_model_artifact()


def load_uploaded_audio(contents: bytes, filename: str | None):
    audio_bytes = io.BytesIO(contents)

    try:
        audio_bytes.seek(0)
        y, sr = sf.read(audio_bytes)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32), sr
    except Exception:
        suffix = os.path.splitext(filename or "")[1] or ".webm"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(contents)
                temp_path = tmp_file.name
            y, sr = librosa.load(temp_path, sr=22050, mono=True)
            return y.astype(np.float32), sr
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def extract_voice_features(y: np.ndarray, sr: int) -> dict:
    features = {}

    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        mask = voiced_flag & ~np.isnan(f0)
        f0_voiced = f0[mask]
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) < 5:
        try:
            f0_yin = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0_voiced = f0_yin[np.isfinite(f0_yin) & (f0_yin > 50)]
        except Exception:
            f0_voiced = np.array([])

    if len(f0_voiced) < 5:
        raise ValueError(
            "Not enough voiced audio detected. Please record at least 5 seconds of clear speech in a quiet room."
        )

    features["MDVP_Fo_Hz"] = float(np.mean(f0_voiced))
    features["MDVP_Fhi_Hz"] = float(np.max(f0_voiced))
    features["MDVP_Flo_Hz"] = float(np.min(f0_voiced))

    f0_diffs = np.abs(np.diff(f0_voiced))
    mean_f0 = float(np.mean(f0_voiced)) + 1e-10
    features["MDVP_Jitter_pct"] = float(np.mean(f0_diffs) / mean_f0 * 100)
    features["MDVP_Jitter_Abs"] = float(np.mean(f0_diffs))
    features["MDVP_RAP"] = float(np.mean(np.abs(np.diff(f0_voiced))) / mean_f0)
    features["MDVP_PPQ"] = float(np.mean(np.abs(f0_voiced - np.roll(f0_voiced, 1))) / mean_f0)
    features["Jitter_DDP"] = features["MDVP_RAP"] * 3

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    rms_nonzero = rms[rms > 1e-8]
    if len(rms_nonzero) < 2:
        rms_nonzero = rms + 1e-8

    amp_diffs = np.abs(np.diff(rms_nonzero))
    mean_rms = float(np.mean(rms_nonzero)) + 1e-10
    features["MDVP_Shimmer"] = float(np.mean(amp_diffs) / mean_rms)
    features["MDVP_Shimmer_dB"] = float(20 * np.log10(1 + features["MDVP_Shimmer"] + 1e-10))
    features["Shimmer_APQ3"] = features["MDVP_Shimmer"]
    features["Shimmer_APQ5"] = features["MDVP_Shimmer"] * 1.1
    features["MDVP_APQ"] = features["MDVP_Shimmer"] * 1.2
    features["Shimmer_DDA"] = features["MDVP_Shimmer"] * 3

    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic
    hnr_power = float(np.mean(harmonic**2)) + 1e-10
    noise_power = float(np.mean(noise**2)) + 1e-10
    features["NHR"] = noise_power / hnr_power
    features["HNR"] = float(10 * np.log10(hnr_power / noise_power))

    seg = y[:2000] if len(y) >= 2000 else y
    autocorr = np.correlate(seg, seg, mode="full")
    autocorr = autocorr[autocorr.size // 2 :]
    autocorr_norm = autocorr / (autocorr[0] + 1e-10)
    lag = min(100, len(autocorr_norm))
    features["RPDE"] = float(np.std(autocorr_norm[:lag]))

    cumsum = np.cumsum(y - np.mean(y))
    features["DFA"] = float(np.std(cumsum) / (len(y) + 1e-10) * 100)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spread1"] = float(-np.mean(spec_bw))
    features["spread2"] = float(np.std(spec_bw))

    nbins = max(2, min(20, len(f0_voiced)))
    hist, _ = np.histogram(f0_voiced, bins=nbins, density=True)
    hist = hist + 1e-10
    features["PPE"] = float(-np.sum(hist * np.log(hist)) / (np.log(nbins) + 1e-10))

    features["D2"] = float(np.log(np.std(y) + 1e-10) / np.log(len(y) + 1))

    return features


def classify_voice(features: dict) -> dict:
    vector = build_model_vector(features, MODEL_ARTIFACT["feature_mapping"])
    probability = float(MODEL_ARTIFACT["model"].predict_proba(vector)[0, 1])
    has_parkinsons = bool(probability >= 0.5)
    severity = severity_from_probability(probability)
    confidence = round(max(probability, 1 - probability) * 100, 1)

    f0_range = features["MDVP_Fhi_Hz"] - features["MDVP_Flo_Hz"]

    return {
        "has_parkinsons": has_parkinsons,
        "severity": severity,
        "severity_info": MODEL_ARTIFACT["risk_bands"][severity],
        "confidence": confidence,
        "raw_score": round(probability * 100, 1),
        "max_score": 100,
        "model": {
            "name": MODEL_ARTIFACT["model_name"],
            "probability": round(probability, 4),
            "artifact_path": str(ARTIFACT_PATH.name),
            "metrics": MODEL_ARTIFACT["metrics"],
        },
        "features": {key: round(float(value), 6) for key, value in features.items()},
        "key_indicators": {
            "jitter_pct": round(features["MDVP_Jitter_pct"], 4),
            "shimmer": round(features["MDVP_Shimmer"], 4),
            "hnr_db": round(features["HNR"], 2),
            "nhr": round(features["NHR"], 4),
            "ppe": round(features["PPE"], 4),
            "f0_range_hz": round(f0_range, 2),
        },
    }


@app.get("/")
def root():
    return {
        "message": "Parkinson's Voice Analysis API",
        "status": "running",
        "model": MODEL_ARTIFACT["model_name"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_ARTIFACT["model_name"],
        "metrics": MODEL_ARTIFACT["metrics"],
    }


@app.post("/retrain")
def retrain_model():
    global MODEL_ARTIFACT
    MODEL_ARTIFACT = train_and_save_model(force=True)
    return {
        "status": "retrained",
        "model_name": MODEL_ARTIFACT["model_name"],
        "metrics": MODEL_ARTIFACT["metrics"],
    }


@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) < 500:
        raise HTTPException(status_code=400, detail="File too small - no audio detected.")

    try:
        y, sr = load_uploaded_audio(contents, file.filename)

        if len(y) < sr * 2:
            raise HTTPException(
                status_code=422,
                detail="Recording too short. Please record at least 5 seconds of natural speech.",
            )

        features = extract_voice_features(y, sr)
        result = classify_voice(features)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        print("TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
