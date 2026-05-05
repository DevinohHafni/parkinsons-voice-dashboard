#!/bin/bash
set -e

echo "============================================"
echo "  NeuroVoice — Parkinson's Voice Analysis  "
echo "============================================"
echo ""

# ── Backend Setup ──────────────────────────────
echo "[1/4] Setting up Python backend..."
cd backend

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.9+ first."
    exit 1
fi

python3 -m venv venv 2>/dev/null || true
source venv/bin/activate

pip install -q -r requirements.txt
echo "✓ Backend dependencies installed"

# ── Start Backend ──────────────────────────────
echo ""
echo "[2/4] Starting FastAPI backend on http://localhost:8000 ..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "✓ Backend PID: $BACKEND_PID"

# Wait for backend to come up
sleep 3

# ── Frontend Setup ─────────────────────────────
echo ""
echo "[3/4] Setting up React frontend..."
cd ../frontend

if ! command -v node &>/dev/null; then
    echo "ERROR: Node.js not found. Install Node.js 18+ first."
    kill $BACKEND_PID
    exit 1
fi

npm install --silent
echo "✓ Frontend dependencies installed"

# ── Start Frontend ─────────────────────────────
echo ""
echo "[4/4] Starting React frontend on http://localhost:5173 ..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "============================================"
echo "  🟢 BOTH SERVERS RUNNING"
echo "  Frontend → http://localhost:5173"
echo "  Backend  → http://localhost:8000"
echo "  API Docs → http://localhost:8000/docs"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers."
echo ""

# Cleanup on exit
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
