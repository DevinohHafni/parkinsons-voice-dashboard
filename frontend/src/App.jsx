import { useState, useRef, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── Color palette ─────────────────────────────────────────────────────────
const COLORS = {
  bg: "#0A0D14",
  surface: "#10141E",
  card: "#141824",
  border: "#1E2433",
  accent: "#4FC3F7",
  accentDim: "#1A3A4A",
  green: "#00C896",
  yellow: "#FFD166",
  orange: "#FF9F43",
  red: "#EE4B6A",
  text: "#E8EAF0",
  muted: "#6B7280",
  subtle: "#9CA3AF",
};

const SEVERITY_COLOR = ["#00C896", "#FFD166", "#FF9F43", "#EE4B6A"];
const SEVERITY_BG = ["#002A1F", "#2A2200", "#2A1800", "#2A0A12"];

// ─── Waveform Visualizer ────────────────────────────────────────────────────
function WaveformVisualizer({ analyserRef, isRecording }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const draw = () => {
      animRef.current = requestAnimationFrame(draw);
      const W = canvas.width, H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      if (!isRecording || !analyserRef.current) {
        // Idle flat line
        ctx.strokeStyle = COLORS.border;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, H / 2);
        ctx.lineTo(W, H / 2);
        ctx.stroke();
        return;
      }

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserRef.current.getByteTimeDomainData(dataArray);

      const grad = ctx.createLinearGradient(0, 0, W, 0);
      grad.addColorStop(0, "#4FC3F7");
      grad.addColorStop(0.5, "#00C896");
      grad.addColorStop(1, "#4FC3F7");
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = "#4FC3F7";
      ctx.shadowBlur = 8;
      ctx.beginPath();

      const sliceWidth = W / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * H) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(W, H / 2);
      ctx.stroke();
      ctx.shadowBlur = 0;
    };

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [isRecording, analyserRef]);

  return (
    <canvas
      ref={canvasRef}
      width={700}
      height={100}
      style={{ width: "100%", height: "100px", borderRadius: "8px" }}
    />
  );
}

// ─── Gauge ──────────────────────────────────────────────────────────────────
function ConfidenceGauge({ value, color }) {
  const r = 54, cx = 70, cy = 70;
  const circumference = Math.PI * r;
  const progress = (value / 100) * circumference;

  return (
    <svg width="140" height="80" viewBox="0 0 140 80">
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke="#1E2433" strokeWidth="10" strokeLinecap="round"
      />
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={`${progress} ${circumference}`}
        style={{ transition: "stroke-dasharray 1s ease, stroke 0.5s" }}
      />
      <text x={cx} y={cy - 8} textAnchor="middle" fill={color}
        style={{ fontSize: "22px", fontWeight: "700", fontFamily: "monospace" }}>
        {value}%
      </text>
      <text x={cx} y={cy + 8} textAnchor="middle" fill={COLORS.muted}
        style={{ fontSize: "10px", fontFamily: "sans-serif" }}>
        CONFIDENCE
      </text>
    </svg>
  );
}

// ─── Feature Bar ─────────────────────────────────────────────────────────────
function FeatureBar({ label, value, max, unit, warn }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = warn
    ? pct > 80 ? COLORS.red : pct > 50 ? COLORS.orange : COLORS.green
    : COLORS.accent;

  return (
    <div style={{ marginBottom: "10px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ fontSize: "11px", color: COLORS.muted, letterSpacing: "0.05em" }}>{label}</span>
        <span style={{ fontSize: "11px", color, fontFamily: "monospace", fontWeight: "600" }}>
          {typeof value === "number" ? value.toFixed(4) : value}{unit}
        </span>
      </div>
      <div style={{ height: "4px", background: COLORS.border, borderRadius: "2px" }}>
        <div style={{
          height: "100%", width: `${pct}%`, background: color,
          borderRadius: "2px", transition: "width 1s ease",
          boxShadow: `0 0 6px ${color}66`
        }} />
      </div>
    </div>
  );
}

// ─── Severity Badge ──────────────────────────────────────────────────────────
function SeverityBadge({ level }) {
  const labels = ["Low", "Mild", "Elevated", "High"];
  const color = SEVERITY_COLOR[level];
  return (
    <span style={{
      display: "inline-block", padding: "3px 10px",
      background: SEVERITY_BG[level], border: `1px solid ${color}`,
      borderRadius: "20px", color, fontSize: "11px", fontWeight: "700",
      letterSpacing: "0.08em"
    }}>
      {labels[level]}
    </span>
  );
}

// ─── Timer ──────────────────────────────────────────────────────────────────
function RecordTimer({ seconds }) {
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  return (
    <span style={{ fontFamily: "monospace", fontSize: "18px", color: COLORS.red, letterSpacing: "0.1em" }}>
      ⏺ {mm}:{ss}
    </span>
  );
}

function mergeBuffers(chunks, totalLength) {
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset, value) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [status, setStatus] = useState("idle"); // idle | recording | analyzing | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [seconds, setSeconds] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);

  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);
  const pcmChunksRef = useRef([]);
  const sampleRateRef = useRef(44100);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);
  const timerRef = useRef(null);
  const streamRef = useRef(null);

  // Timer
  useEffect(() => {
    if (status === "recording") {
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds(s => s + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [status]);

  const startRecording = useCallback(async () => {
    try {
      setError("");
      setResult(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Web Audio for visualiser
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      sampleRateRef.current = audioCtxRef.current.sampleRate;
      const source = audioCtxRef.current.createMediaStreamSource(stream);
      sourceRef.current = source;
      const analyser = audioCtxRef.current.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      const processor = audioCtxRef.current.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      pcmChunksRef.current = [];
      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(input));
      };

      source.connect(processor);
      processor.connect(audioCtxRef.current.destination);
      setStatus("recording");
    } catch (e) {
      setError("Microphone access denied. Please allow microphone permissions and try again.");
      setStatus("error");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (status === "recording") {
      const totalLength = pcmChunksRef.current.reduce((sum, chunk) => sum + chunk.length, 0);
      const merged = mergeBuffers(pcmChunksRef.current, totalLength);
      const blob = encodeWav(merged, sampleRateRef.current);

      processorRef.current?.disconnect();
      sourceRef.current?.disconnect();
      streamRef.current?.getTracks().forEach(t => t.stop());
      audioCtxRef.current?.close();
      analyserRef.current = null;
      setAudioBlob(blob);
      setStatus("analyzing");
      analyseBlob(blob);
    }
  }, [status]);

  const analyseBlob = async (blob) => {
    try {
      setStatus("analyzing");
      const formData = new FormData();
      formData.append("file", blob, "recording.wav");
      const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: formData });
      if (!res.ok) {
        let message = "Analysis failed";
        try {
          const err = await res.json();
          message = err.detail || err.message || message;
        } catch {
          message = await res.text() || message;
        }
        throw new Error(message);
      }
      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (e) {
      setError(e.message || "Could not connect to analysis server. Make sure the backend is running.");
      setStatus("error");
    }
  };

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setError("");
    setSeconds(0);
    setAudioBlob(null);
  };

  const sev = result?.severity ?? 0;
  const sevColor = SEVERITY_COLOR[sev];

  return (
    <div style={{
      minHeight: "100vh", background: COLORS.bg, color: COLORS.text,
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      padding: "0", margin: "0"
    }}>
      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "18px 32px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: COLORS.surface,
        position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <div style={{
            width: "36px", height: "36px", borderRadius: "8px",
            background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.green})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "18px"
          }}>🧠</div>
          <div>
            <div style={{ fontSize: "14px", fontWeight: "700", letterSpacing: "0.12em", color: COLORS.text }}>
              NEUROVOICE
            </div>
            <div style={{ fontSize: "10px", color: COLORS.muted, letterSpacing: "0.08em" }}>
              PARKINSON'S VOICE ANALYSIS SYSTEM
            </div>
          </div>
        </div>

      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "32px 24px" }}>

        {/* Hero Instruction */}
        <div style={{
          textAlign: "center", marginBottom: "40px"
        }}>
          <h1 style={{
            fontSize: "clamp(24px, 4vw, 42px)", fontWeight: "800",
            letterSpacing: "-0.02em", lineHeight: 1.1,
            background: `linear-gradient(135deg, ${COLORS.text} 40%, ${COLORS.accent})`,
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            fontFamily: "'IBM Plex Mono', monospace", margin: "0 0 12px"
          }}>
            Voice-Based Parkinson's Detection
          </h1>
          <p style={{ color: COLORS.muted, fontSize: "14px", maxWidth: "560px", margin: "0 auto", lineHeight: 1.7 }}>
            Speak naturally — say your name, today's date, or read a sentence aloud.
            Our system analyzes 22 vocal biomarkers from your natural speech and runs them through a trained Parkinson's voice classifier.
          </p>
          <div style={{
            display: "inline-block", marginTop: "12px",
            padding: "6px 16px", background: "#1A1000",
            border: "1px solid #3A2800", borderRadius: "20px",
            fontSize: "11px", color: "#FFD166", letterSpacing: "0.06em"
          }}>
            ⚠ FOR SCREENING PURPOSES ONLY — NOT A MEDICAL DIAGNOSIS
          </div>
        </div>

        {/* Recording Card */}
        <div style={{
          background: COLORS.card,
          border: `1px solid ${status === "recording" ? COLORS.red + "66" : COLORS.border}`,
          borderRadius: "16px", padding: "32px",
          marginBottom: "28px",
          boxShadow: status === "recording" ? `0 0 30px ${COLORS.red}22` : "none",
          transition: "all 0.3s"
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "24px" }}>
            <span style={{ fontSize: "11px", color: COLORS.muted, letterSpacing: "0.1em" }}>VOICE INPUT MODULE</span>
            {status === "recording" && <RecordTimer seconds={seconds} />}
          </div>

          {/* Waveform */}
          <div style={{
            background: COLORS.bg, borderRadius: "10px", padding: "12px",
            marginBottom: "24px", border: `1px solid ${COLORS.border}`
          }}>
            <WaveformVisualizer analyserRef={analyserRef} isRecording={status === "recording"} />
          </div>

          {/* Controls */}
          <div style={{ display: "flex", gap: "14px", justifyContent: "center", alignItems: "center" }}>
            {status === "idle" || status === "done" || status === "error" ? (
              <>
                {(status === "done" || status === "error") && (
                  <button onClick={reset} style={{
                    padding: "12px 28px", borderRadius: "8px", border: `1px solid ${COLORS.border}`,
                    background: "transparent", color: COLORS.subtle, cursor: "pointer", fontSize: "13px",
                    letterSpacing: "0.06em", transition: "all 0.2s"
                  }}
                    onMouseOver={e => e.target.style.borderColor = COLORS.accent}
                    onMouseOut={e => e.target.style.borderColor = COLORS.border}
                  >
                    ↺ RESET
                  </button>
                )}
                <button onClick={startRecording} style={{
                  padding: "14px 40px", borderRadius: "8px",
                  background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.green})`,
                  border: "none", color: "#000", cursor: "pointer",
                  fontSize: "13px", fontWeight: "700", letterSpacing: "0.1em",
                  boxShadow: `0 4px 20px ${COLORS.accent}44`,
                  transition: "all 0.2s",
                  fontFamily: "'IBM Plex Mono', monospace"
                }}>
                  ▶ START RECORDING
                </button>
              </>
            ) : status === "recording" ? (
              <button onClick={stopRecording} style={{
                padding: "14px 40px", borderRadius: "8px",
                background: COLORS.red, border: "none",
                color: "#fff", cursor: "pointer",
                fontSize: "13px", fontWeight: "700", letterSpacing: "0.1em",
                boxShadow: `0 4px 20px ${COLORS.red}66`,
                animation: "pulse 1.5s infinite",
                fontFamily: "'IBM Plex Mono', monospace"
              }}>
                ■ STOP & ANALYZE
              </button>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
                <div style={{
                  width: "40px", height: "40px", border: `3px solid ${COLORS.accent}`,
                  borderTop: "3px solid transparent", borderRadius: "50%",
                  animation: "spin 1s linear infinite"
                }} />
                <span style={{ color: COLORS.accent, fontSize: "12px", letterSpacing: "0.1em" }}>
                  ANALYZING VOICE BIOMARKERS...
                </span>
              </div>
            )}
          </div>

          {error && (
            <div style={{
              marginTop: "20px", padding: "12px 16px",
              background: "#2A0A12", border: "1px solid #EE4B6A55",
              borderRadius: "8px", color: COLORS.red, fontSize: "13px"
            }}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* Results */}
        {result && (
          <div style={{ animation: "fadeIn 0.6s ease" }}>

            {/* Main Result */}
            <div style={{
              background: SEVERITY_BG[sev],
              border: `1px solid ${sevColor}55`,
              borderRadius: "16px", padding: "28px",
              marginBottom: "24px",
              boxShadow: `0 0 40px ${sevColor}22`
            }}>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "20px" }}>
                <div>
                  <div style={{ fontSize: "10px", color: COLORS.muted, letterSpacing: "0.12em", marginBottom: "10px" }}>
                    DIAGNOSIS RESULT
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "14px", marginBottom: "10px" }}>
                    <div style={{ fontSize: "40px" }}>{result.has_parkinsons ? "🔴" : "🟢"}</div>
                    <div>
                      <div style={{ fontSize: "clamp(18px, 3vw, 26px)", fontWeight: "800", color: sevColor, lineHeight: 1.1 }}>
                        {result.severity_info.label}
                      </div>
                      <div style={{ marginTop: "6px" }}>
                        <SeverityBadge level={sev} />
                      </div>
                    </div>
                  </div>
                  <p style={{ color: COLORS.subtle, fontSize: "13px", lineHeight: 1.7, maxWidth: "480px", margin: "12px 0 0" }}>
                    {result.severity_info.description}
                  </p>
                </div>
                <ConfidenceGauge value={result.confidence} color={sevColor} />
              </div>

              <div style={{
                marginTop: "20px", padding: "14px 18px",
                background: `${sevColor}11`, border: `1px solid ${sevColor}33`,
                borderRadius: "8px", fontSize: "13px", color: COLORS.text, lineHeight: 1.7
              }}>
                💬 <strong>Recommendation:</strong> {result.severity_info.recommendation}
              </div>
            </div>

            {/* Two-column: Key Indicators + Full Features */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "20px" }}>

              {/* Key Indicators */}
              <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: "14px", padding: "24px" }}>
                <div style={{ fontSize: "11px", color: COLORS.muted, letterSpacing: "0.1em", marginBottom: "18px" }}>
                  KEY VOICE BIOMARKERS
                </div>
                <FeatureBar label="MDVP:Jitter(%)" value={result.key_indicators.jitter_pct} max={5} unit="%" warn />
                <FeatureBar label="MDVP:Shimmer" value={result.key_indicators.shimmer} max={0.3} unit="" warn />
                <FeatureBar label="HNR (dB)" value={Math.max(0, 30 - result.key_indicators.hnr_db)} max={30} unit="" warn />
                <FeatureBar label="NHR" value={result.key_indicators.nhr} max={0.5} unit="" warn />
                <FeatureBar label="PPE" value={result.key_indicators.ppe} max={1} unit="" warn />
                <FeatureBar label="F0 Range (Hz)" value={Math.max(0, 150 - result.key_indicators.f0_range_hz)} max={150} unit="" warn />

                {/* Score bar */}
                <div style={{ marginTop: "20px", paddingTop: "16px", borderTop: `1px solid ${COLORS.border}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                    <span style={{ fontSize: "11px", color: COLORS.muted }}>COMPOSITE RISK SCORE</span>
                    <span style={{ fontSize: "13px", color: sevColor, fontWeight: "700" }}>
                      {result.raw_score} / {result.max_score}
                    </span>
                  </div>
                  <div style={{ height: "8px", background: COLORS.border, borderRadius: "4px" }}>
                    <div style={{
                      height: "100%",
                      width: `${(result.raw_score / result.max_score) * 100}%`,
                      background: `linear-gradient(90deg, ${COLORS.green}, ${sevColor})`,
                      borderRadius: "4px", transition: "width 1.2s ease",
                      boxShadow: `0 0 10px ${sevColor}66`
                    }} />
                  </div>
                </div>
              </div>

              {/* All 22 Features */}
              <div style={{ background: COLORS.card, border: `1px solid ${COLORS.border}`, borderRadius: "14px", padding: "24px" }}>
                <div style={{ fontSize: "11px", color: COLORS.muted, letterSpacing: "0.1em", marginBottom: "18px" }}>
                  ALL EXTRACTED FEATURES (22)
                </div>
                <div style={{ maxHeight: "360px", overflowY: "auto" }}>
                  {Object.entries(result.features).map(([k, v]) => (
                    <div key={k} style={{
                      display: "flex", justifyContent: "space-between", alignItems: "center",
                      padding: "6px 0", borderBottom: `1px solid ${COLORS.border}`
                    }}>
                      <span style={{ fontSize: "11px", color: COLORS.muted }}>{k.replace(/_/g, " ")}</span>
                      <span style={{ fontSize: "12px", color: COLORS.accent, fontFamily: "monospace" }}>
                        {typeof v === "number" ? v.toFixed(5) : v}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Risk band visual */}
            <div style={{
              background: COLORS.card, border: `1px solid ${COLORS.border}`,
              borderRadius: "14px", padding: "24px", marginTop: "20px"
            }}>
              <div style={{ fontSize: "11px", color: COLORS.muted, letterSpacing: "0.1em", marginBottom: "18px" }}>
                MODEL RISK BANDS
              </div>
              <div style={{ display: "flex", gap: "0", position: "relative" }}>
                {[
                  { label: "Low", sub: "<35%", color: COLORS.green },
                  { label: "Mild", sub: "35-55%", color: COLORS.yellow },
                  { label: "Elevated", sub: "55-75%", color: COLORS.orange },
                  { label: "High", sub: "75%+", color: COLORS.red },
                ].map((s, i) => (
                  <div key={i} style={{
                    flex: 1, textAlign: "center", padding: "14px 8px",
                    background: i === sev ? `${s.color}22` : "transparent",
                    border: `1px solid ${i === sev ? s.color : COLORS.border}`,
                    borderRadius: i === 0 ? "8px 0 0 8px" : i === 3 ? "0 8px 8px 0" : "0",
                    transition: "all 0.5s",
                    boxShadow: i === sev ? `0 0 20px ${s.color}44` : "none"
                  }}>
                    <div style={{ fontSize: "13px", fontWeight: "700", color: i === sev ? s.color : COLORS.muted }}>
                      {s.label}
                    </div>
                    <div style={{ fontSize: "10px", color: COLORS.muted, marginTop: "4px" }}>{s.sub}</div>
                    {i === sev && (
                      <div style={{ fontSize: "16px", marginTop: "6px" }}>▲</div>
                    )}
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: "48px", textAlign: "center", color: COLORS.muted, fontSize: "11px", lineHeight: 1.8 }}>
          <div>Built for screening research. Uses MDVP-equivalent feature extraction plus a trained classifier based on the Oxford Parkinson's dataset.</div>
          <div>⚠ Not a substitute for professional medical evaluation.</div>
        </div>

      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: ${COLORS.bg}; }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse {
          0%, 100% { box-shadow: 0 4px 20px ${COLORS.red}66; }
          50% { box-shadow: 0 4px 40px ${COLORS.red}; }
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${COLORS.bg}; }
        ::-webkit-scrollbar-thumb { background: ${COLORS.border}; border-radius: 3px; }
      `}</style>
    </div>
  );
}
