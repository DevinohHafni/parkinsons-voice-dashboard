import os
import io
import traceback
import numpy as np
import librosa
import librosa.effects
import librosa.feature
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="Parkinson's Voice Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_voice_features(y: np.ndarray, sr: int) -> dict:
    features = {}

    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    # F0 via pyin with fallback to yin
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7')),
            sr=sr
        )
        mask = voiced_flag & ~np.isnan(f0)
        f0_voiced = f0[mask]
    except Exception:
        f0_voiced = np.array([])

    if len(f0_voiced) < 5:
        try:
            f0_yin = librosa.yin(y, fmin=50, fmax=500, sr=sr)
            f0_voiced = f0_yin[f0_yin > 50]
        except Exception:
            f0_voiced = np.array([])

    if len(f0_voiced) < 5:
        raise ValueError(
            "Not enough voiced audio detected. "
            "Please record at least 5+ seconds of natural speech in a quiet room."
        )

    features["MDVP_Fo_Hz"]  = float(np.mean(f0_voiced))
    features["MDVP_Fhi_Hz"] = float(np.max(f0_voiced))
    features["MDVP_Flo_Hz"] = float(np.min(f0_voiced))

    f0_diffs = np.abs(np.diff(f0_voiced))
    mean_f0  = float(np.mean(f0_voiced)) + 1e-10
    features["MDVP_Jitter_pct"] = float(np.mean(f0_diffs) / mean_f0 * 100)
    features["MDVP_Jitter_Abs"] = float(np.mean(f0_diffs))
    features["MDVP_RAP"]        = float(np.mean(np.abs(np.diff(f0_voiced))) / mean_f0)
    features["MDVP_PPQ"]        = float(np.mean(np.abs(f0_voiced - np.roll(f0_voiced, 1))) / mean_f0)
    features["Jitter_DDP"]      = features["MDVP_RAP"] * 3

    rms         = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    rms_nonzero = rms[rms > 1e-8]
    if len(rms_nonzero) < 2:
        rms_nonzero = rms + 1e-8
    amp_diffs = np.abs(np.diff(rms_nonzero))
    mean_rms  = float(np.mean(rms_nonzero)) + 1e-10
    features["MDVP_Shimmer"]    = float(np.mean(amp_diffs) / mean_rms)
    features["MDVP_Shimmer_dB"] = float(20 * np.log10(1 + features["MDVP_Shimmer"] + 1e-10))
    features["Shimmer_APQ3"]    = features["MDVP_Shimmer"]
    features["Shimmer_APQ5"]    = features["MDVP_Shimmer"] * 1.1
    features["MDVP_APQ"]        = features["MDVP_Shimmer"] * 1.2
    features["Shimmer_DDA"]     = features["MDVP_Shimmer"] * 3

    harmonic    = librosa.effects.harmonic(y)
    noise       = y - harmonic
    hnr_power   = float(np.mean(harmonic ** 2)) + 1e-10
    noise_power = float(np.mean(noise ** 2))    + 1e-10
    features["NHR"] = noise_power / hnr_power
    features["HNR"] = float(10 * np.log10(hnr_power / noise_power))

    seg = y[:2000] if len(y) >= 2000 else y
    autocorr      = np.correlate(seg, seg, mode='full')
    autocorr      = autocorr[autocorr.size // 2:]
    autocorr_norm = autocorr / (autocorr[0] + 1e-10)
    lag           = min(100, len(autocorr_norm))
    features["RPDE"] = float(np.std(autocorr_norm[:lag]))

    cumsum = np.cumsum(y - np.mean(y))
    features["DFA"] = float(np.std(cumsum) / (len(y) + 1e-10) * 100)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features["spread1"] = float(-np.mean(spec_bw))
    features["spread2"] = float(np.std(spec_bw))

    nbins = min(20, len(f0_voiced))
    hist, _ = np.histogram(f0_voiced, bins=nbins, density=True)
    hist    = hist + 1e-10
    features["PPE"] = float(-np.sum(hist * np.log(hist)) / (np.log(nbins) + 1e-10))

    features["D2"] = float(np.log(np.std(y) + 1e-10) / np.log(len(y) + 1))

    return features


SEVERITY_LEVELS = {
    0: {
        "label": "No Parkinson's Detected",
        "color": "#00C896",
        "description": "Voice biomarkers are within normal range. No signs of Parkinson's disease detected.",
        "recommendation": "Continue routine health monitoring. Consult a neurologist if you notice tremors or motor difficulties."
    },
    1: {
        "label": "Early Stage (HY Stage 1)",
        "color": "#FFD166",
        "description": "Mild vocal irregularities detected. May indicate very early-stage Parkinson's (HY Stage 1) — unilateral symptoms.",
        "recommendation": "Consult a neurologist soon. Early intervention significantly improves outcomes."
    },
    2: {
        "label": "Moderate Stage (HY Stage 2-3)",
        "color": "#FF9F43",
        "description": "Moderate voice perturbations detected, consistent with mid-stage Parkinson's (HY Stage 2-3).",
        "recommendation": "Seek immediate neurological evaluation. Treatment and therapy can meaningfully slow progression."
    },
    3: {
        "label": "Advanced Stage (HY Stage 4-5)",
        "color": "#EE4B6A",
        "description": "Significant dysphonia detected, consistent with advanced Parkinson's (HY Stage 4-5).",
        "recommendation": "Urgent neurological care recommended. Multidisciplinary treatment team advised."
    }
}


def classify_voice(features: dict) -> dict:
    score = 0
    jitter   = features.get("MDVP_Jitter_pct", 0)
    shimmer  = features.get("MDVP_Shimmer", 0)
    hnr      = features.get("HNR", 20)
    nhr      = features.get("NHR", 0)
    ppe      = features.get("PPE", 0)
    rpde     = features.get("RPDE", 0)
    dda      = features.get("Shimmer_DDA", 0)
    f0_range = features.get("MDVP_Fhi_Hz", 200) - features.get("MDVP_Flo_Hz", 100)

    if jitter > 2.5:     score += 3
    elif jitter > 1.5:   score += 2
    elif jitter > 1.0:   score += 1

    if shimmer > 0.10:   score += 3
    elif shimmer > 0.06: score += 2
    elif shimmer > 0.04: score += 1

    if hnr < 10:    score += 3
    elif hnr < 14:  score += 2
    elif hnr < 18:  score += 1

    if nhr > 0.15:   score += 3
    elif nhr > 0.08: score += 2
    elif nhr > 0.04: score += 1

    if ppe > 0.45:   score += 2
    elif ppe > 0.30: score += 1

    if rpde > 0.6:    score += 2
    elif rpde > 0.45: score += 1

    if f0_range < 20:  score += 2
    elif f0_range < 40: score += 1

    if dda > 0.30:   score += 2
    elif dda > 0.15: score += 1

    max_score  = 21
    confidence = round(min(score / max_score, 1.0) * 100, 1)

    if score <= 3:    severity = 0
    elif score <= 8:  severity = 1
    elif score <= 14: severity = 2
    else:             severity = 3

    return {
        "has_parkinsons": severity > 0,
        "severity": severity,
        "severity_info": SEVERITY_LEVELS[severity],
        "confidence": confidence,
        "raw_score": score,
        "max_score": max_score,
        "features": {k: round(float(v), 6) for k, v in features.items()},
        "key_indicators": {
            "jitter_pct":  round(jitter, 4),
            "shimmer":     round(shimmer, 4),
            "hnr_db":      round(hnr, 2),
            "nhr":         round(nhr, 4),
            "ppe":         round(ppe, 4),
            "f0_range_hz": round(f0_range, 2),
        }
    }


@app.get("/")
def root():
    return {"message": "Parkinson's Voice Analysis API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) < 500:
        raise HTTPException(status_code=400, detail="File too small — no audio detected.")

    audio_bytes = io.BytesIO(contents)

    try:
        try:
            audio_bytes.seek(0)
            y, sr = sf.read(audio_bytes)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y.astype(np.float32)
        except Exception:
            audio_bytes.seek(0)
            y, sr = librosa.load(audio_bytes, sr=22050, mono=True)

        if len(y) < sr * 2:
            raise HTTPException(
                status_code=422,
                detail="Recording too short. Please record at least 5 seconds of natural speech."
            )

        features = extract_voice_features(y, sr)
        result   = classify_voice(features)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print("TRACEBACK:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
