# 🧠 NeuroVoice — Parkinson's Voice Analysis System

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **A full-stack AI-powered dashboard that analyzes natural speech to screen for Parkinson's disease indicators — in real time.**

Simply speak into your mic. NeuroVoice extracts 22 clinical vocal biomarkers from your voice and classifies the likelihood and severity of Parkinson's disease using the Hoehn & Yahr scale.

⚠️ *For screening and research purposes only — not a substitute for professional medical diagnosis.*

---

## 📸 Preview

```
🎙️ Speak naturally → 🧠 22 biomarkers extracted → 📊 Severity classified (HY 1–5)
```

---

## ✨ Features

- 🎙️ **Real-time voice recording** with live waveform visualizer
- 🔬 **22 MDVP-equivalent biomarkers** extracted (Jitter, Shimmer, HNR, NHR, PPE, RPDE, DFA, D2...)
- 📊 **4-level severity classification** using the Hoehn & Yahr scale
- 🩺 **Confidence score** and composite risk scoring
- 💬 **Natural speech** — just say your name or read a sentence, no sustained vowels needed
- 🌙 **Dark mode** clinical dashboard UI
- ⚡ **FastAPI backend** with librosa-based signal processing
- 🔓 **Fully open source** — fork, extend, improve

---

## 🏗️ Tech Stack

| Layer | Tech |
|-------|------|
| Frontend | React 18 + Vite |
| Backend | FastAPI + Uvicorn |
| Audio DSP | Librosa, SoundFile, NumPy |
| ML/Scoring | Rule-based classifier (calibrated to Oxford PD Dataset) |
| Styling | Pure CSS-in-JS, IBM Plex Mono |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+

### 1. Clone the repo

```bash
git clone https://github.com/DevinohHafni/parkinsons-voice-dashboard.git
cd parkinsons-voice-dashboard
```

### 2. Start the Backend

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Backend runs at → `http://localhost:8000`
API docs at → `http://localhost:8000/docs`

### 3. Start the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at → `http://localhost:5173`

---

## 🎯 How to Use

1. Open `http://localhost:5173`
2. Click **START RECORDING**
3. Speak naturally — say your name, today's date, or read a sentence (5–10 seconds)
4. Click **STOP & ANALYZE**
5. View your results with severity level, confidence %, and all 22 biomarkers

---

## 📁 Project Structure

```
parkinsons-voice-dashboard/
├── backend/
│   ├── main.py              # FastAPI server + feature extraction + classifier
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Full dashboard UI
│   │   └── main.jsx         # React entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 🔬 Biomarkers Extracted

| Category | Features |
|----------|----------|
| Fundamental Frequency | MDVP:Fo, MDVP:Fhi, MDVP:Flo |
| Jitter | MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP |
| Shimmer | MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA |
| Noise Ratios | NHR, HNR |
| Nonlinear Dynamics | RPDE, DFA, D2 |
| Spectral | spread1, spread2, PPE |

---

## 🧩 Severity Classification

| Score | Level | Hoehn & Yahr |
|-------|-------|--------------|
| 0–3 | ✅ No Parkinson's | — |
| 4–8 | 🟡 Early Stage | HY Stage 1 |
| 9–14 | 🟠 Moderate | HY Stage 2–3 |
| 15+ | 🔴 Advanced | HY Stage 4–5 |

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# 1. Fork this repo (click Fork button top right)
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/parkinsons-voice-dashboard.git

# 3. Create a branch
git checkout -b feature/your-feature-name

# 4. Make your changes, then commit
git add .
git commit -m "Add: your feature description"

# 5. Push and open a Pull Request
git push origin feature/your-feature-name
```

### Ideas for contributions
- [ ] Train a proper ML model on the Oxford PD Dataset
- [ ] Add patient history / session logging
- [ ] Export results as PDF report
- [ ] Add multilingual support
- [ ] Mobile app version
- [ ] Upload audio file instead of live recording
- [ ] Add tremor detection via webcam

---

## 📄 License

MIT License — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Oxford Parkinson's Disease Detection Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons) — Max Little, University of Oxford
- [Librosa](https://librosa.org/) — audio analysis library
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python web framework

---

<p align="center">Built with ❤️ for early disease detection research</p>
