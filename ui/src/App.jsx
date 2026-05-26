import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "https://intent-classifier-api-8tte.onrender.com";

const INTENT_META = {
  refund_request:       { icon: "💸", color: "#b45309", bg: "#fef3c7", label: "Refund Request" },
  payment_failed:       { icon: "❌", color: "#b91c1c", bg: "#fee2e2", label: "Payment Failed" },
  account_suspended:    { icon: "🔒", color: "#6d28d9", bg: "#ede9fe", label: "Account Suspended" },
  api_integration_help: { icon: "⚙️", color: "#1d4ed8", bg: "#dbeafe", label: "API Help" },
  subscription_upgrade: { icon: "⬆️", color: "#065f46", bg: "#d1fae5", label: "Upgrade Plan" },
  subscription_cancel:  { icon: "🚫", color: "#9a3412", bg: "#ffedd5", label: "Cancel Plan" },
  security_concern:     { icon: "🛡️", color: "#991b1b", bg: "#fce7f3", label: "Security Concern" },
  general_inquiry:      { icon: "💬", color: "#374151", bg: "#f3f4f6", label: "General Inquiry" },
};

const SAMPLES = [
  "I want a refund for my last payment",
  "My card keeps getting declined",
  "How do I authenticate API requests?",
  "I think someone hacked my account",
  "Please cancel my subscription",
  "I want to upgrade to the Pro plan",
  "My account has been suspended",
  "What currencies do you support?",
];

function ConfidenceBar({ intent, confidence, color, isTop }) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => setWidth(confidence * 100), 50);
    return () => clearTimeout(t);
  }, [confidence]);

  const meta = INTENT_META[intent] || {};

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: "10px",
      padding: "6px 0",
      borderBottom: "1px solid #f1f5f9",
    }}>
      <div style={{ width: "160px", fontSize: "12px", color: isTop ? "#0f172a" : "#64748b", fontWeight: isTop ? 600 : 400, flexShrink: 0 }}>
        {meta.label || intent.replace(/_/g, " ")}
      </div>
      <div style={{ flex: 1, height: "6px", background: "#f1f5f9", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${width}%`,
          background: isTop ? color : "#cbd5e1",
          borderRadius: "3px",
          transition: "width 0.5s cubic-bezier(0.4, 0, 0.2, 1)",
        }} />
      </div>
      <div style={{ width: "40px", textAlign: "right", fontSize: "12px", color: isTop ? "#0f172a" : "#94a3b8", fontWeight: isTop ? 600 : 400 }}>
        {(confidence * 100).toFixed(1)}%
      </div>
    </div>
  );
}

function TierBadge({ tier }) {
  const styles = {
    high:   { bg: "#dcfce7", color: "#166534", label: "High confidence" },
    medium: { bg: "#fef9c3", color: "#854d0e", label: "Medium confidence" },
    low:    { bg: "#fee2e2", color: "#991b1b", label: "Low confidence — review needed" },
  };
  const s = styles[tier] || styles.low;
  return (
    <span style={{
      background: s.bg, color: s.color,
      fontSize: "11px", fontWeight: 600,
      padding: "3px 8px", borderRadius: "999px",
      letterSpacing: "0.03em",
    }}>
      {s.label}
    </span>
  );
}

function TopIntentCard({ result }) {
  const { top_intent, confidence_tier, inference_ms } = result;
  const meta = INTENT_META[top_intent.intent] || {};

  return (
    <div style={{
      background: meta.bg || "#f8fafc",
      border: `1.5px solid ${meta.color}30`,
      borderRadius: "16px",
      padding: "20px 24px",
      marginBottom: "16px",
      transition: "all 0.3s ease",
    }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "12px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{
            width: "44px", height: "44px", borderRadius: "12px",
            background: "white", display: "flex", alignItems: "center",
            justifyContent: "center", fontSize: "22px",
            boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
          }}>
            {meta.icon}
          </div>
          <div>
            <div style={{ fontSize: "18px", fontWeight: 700, color: meta.color, fontFamily: "'DM Serif Display', Georgia, serif" }}>
              {meta.label || top_intent.intent.replace(/_/g, " ")}
            </div>
            <div style={{ fontSize: "12px", color: "#64748b", marginTop: "2px" }}>
              {top_intent.description}
            </div>
          </div>
        </div>
        <div style={{ textAlign: "right", flexShrink: 0 }}>
          <div style={{ fontSize: "28px", fontWeight: 800, color: meta.color, lineHeight: 1 }}>
            {(top_intent.confidence * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "2px" }}>confidence</div>
        </div>
      </div>

      <div style={{ marginTop: "14px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ fontSize: "12px", color: "#64748b" }}>Route to:</span>
          <span style={{
            background: "white", color: meta.color,
            fontSize: "12px", fontWeight: 600,
            padding: "3px 10px", borderRadius: "999px",
            border: `1px solid ${meta.color}40`,
          }}>
            → {top_intent.route_to}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <TierBadge tier={confidence_tier} />
          <span style={{ fontSize: "11px", color: "#94a3b8" }}>{inference_ms}ms</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiOnline, setApiOnline] = useState(null);
  const debounceRef = useRef(null);

  // Check API health on mount
  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  const classify = useCallback(async (input) => {
    if (!input.trim() || input.trim().length < 3) { setResult(null); return; }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input.trim(), top_k: 8 }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleInput = (e) => {
    const val = e.target.value;
    setText(val);
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => classify(val), 220);
  };

  const handleSample = (s) => {
    setText(s);
    classify(s);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#f8fafc",
      fontFamily: "'DM Sans', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        background: "white",
        borderBottom: "1px solid #e2e8f0",
        padding: "0 24px",
      }}>
        <div style={{ maxWidth: "780px", margin: "0 auto", padding: "16px 0", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div>
            <div style={{ fontSize: "20px", fontWeight: 700, color: "#0f172a", fontFamily: "'DM Serif Display', Georgia, serif", letterSpacing: "-0.3px" }}>
              Intent Classifier
            </div>
            <div style={{ fontSize: "12px", color: "#94a3b8", marginTop: "1px" }}>Real-time support ticket routing</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <div style={{
              width: "8px", height: "8px", borderRadius: "50%",
              background: apiOnline === true ? "#22c55e" : apiOnline === false ? "#ef4444" : "#94a3b8",
            }} />
            <span style={{ fontSize: "12px", color: "#64748b" }}>
              {apiOnline === true ? "API connected" : apiOnline === false ? "API offline" : "Connecting..."}
            </span>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: "780px", margin: "0 auto", padding: "32px 24px" }}>

        {/* API offline warning */}
        {apiOnline === false && (
          <div style={{
            background: "#fef2f2", border: "1px solid #fecaca",
            borderRadius: "12px", padding: "14px 18px", marginBottom: "20px",
            fontSize: "13px", color: "#991b1b",
          }}>
            <strong>API server is offline.</strong> Run <code style={{ background: "#fee2e2", padding: "1px 5px", borderRadius: "4px" }}>uvicorn api.main:app --reload --port 8000</code> in your terminal.
          </div>
        )}

        {/* Input */}
        <div style={{
          background: "white", borderRadius: "16px",
          border: "1px solid #e2e8f0", padding: "20px",
          marginBottom: "16px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}>
          <label style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Support message
          </label>
          <textarea
            value={text}
            onChange={handleInput}
            placeholder="Type a user message... e.g. I want to cancel my subscription"
            maxLength={300}
            style={{
              width: "100%", marginTop: "10px",
              minHeight: "80px", padding: "12px",
              fontSize: "15px", lineHeight: "1.6",
              border: "1px solid #e2e8f0", borderRadius: "10px",
              background: "#f8fafc", color: "#0f172a",
              resize: "vertical", outline: "none",
              fontFamily: "inherit",
              transition: "border-color 0.15s",
            }}
            onFocus={e => e.target.style.borderColor = "#94a3b8"}
            onBlur={e => e.target.style.borderColor = "#e2e8f0"}
          />
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "8px" }}>
            <span style={{ fontSize: "11px", color: "#94a3b8" }}>{text.length} / 300 characters</span>
            {loading && <span style={{ fontSize: "11px", color: "#94a3b8" }}>Classifying...</span>}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #fecaca", borderRadius: "10px", padding: "12px 16px", marginBottom: "16px", fontSize: "13px", color: "#991b1b" }}>
            {error} — is the API server running?
          </div>
        )}

        {/* Result */}
        {result && !error && (
          <div style={{ animation: "fadeIn 0.2s ease" }}>
            <TopIntentCard result={result} />

            {/* All intent bars */}
            <div style={{
              background: "white", borderRadius: "16px",
              border: "1px solid #e2e8f0", padding: "20px",
              marginBottom: "16px",
              boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
            }}>
              <div style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "12px" }}>
                All intent probabilities
              </div>
              {result.all_intents.map((item, i) => (
                <ConfidenceBar
                  key={item.intent}
                  intent={item.intent}
                  confidence={item.confidence}
                  color={INTENT_META[item.intent]?.color || "#374151"}
                  isTop={i === 0}
                />
              ))}
            </div>
          </div>
        )}

        {/* Samples */}
        <div style={{
          background: "white", borderRadius: "16px",
          border: "1px solid #e2e8f0", padding: "20px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "#64748b", letterSpacing: "0.05em", textTransform: "uppercase", marginBottom: "12px" }}>
            Try an example
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {SAMPLES.map(s => (
              <button
                key={s}
                onClick={() => handleSample(s)}
                style={{
                  fontSize: "12px", padding: "6px 12px",
                  border: "1px solid #e2e8f0", borderRadius: "999px",
                  background: text === s ? "#0f172a" : "white",
                  color: text === s ? "white" : "#475569",
                  cursor: "pointer", transition: "all 0.15s",
                  fontFamily: "inherit",
                }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div style={{ textAlign: "center", marginTop: "24px", fontSize: "11px", color: "#cbd5e1" }}>
          Powered by FastAPI + scikit-learn · {INTENT_META ? Object.keys(INTENT_META).length : 8} intent classes
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Serif+Display&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        textarea::placeholder { color: #94a3b8; }
        button:hover { background: #f1f5f9 !important; color: #0f172a !important; }
      `}</style>
    </div>
  );
}