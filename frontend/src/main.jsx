import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  CircleStop,
  Cpu,
  Gauge,
  Keyboard,
  MousePointer2,
  Play,
  RotateCcw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Waves,
} from "lucide-react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_NEUROCORE_API_BASE || "http://127.0.0.1:8010";

const defaultSettings = {
  device: {
    name: "generic-eeg",
    sampling_rate: 250,
    channels: ["Fz", "Cz", "Pz", "Oz"],
    channel_aliases: {},
    reference: "common",
  },
  signal: {
    highpass_hz: 1,
    lowpass_hz: 40,
    target_sampling_rate: 250,
    window_seconds: 1,
    step_seconds: 0.25,
    max_clock_drift_ppm: 100,
  },
  safety: {
    min_confidence: 0.75,
    max_actions_per_second: 4,
    require_human_arm: true,
    human_armed: false,
    emergency_stop: false,
    block_prompt_like_agent_payloads: true,
  },
  agent: {
    allowed_tools: ["open_task", "summarize_context", "draft_reply", "run_local_plan"],
    max_payload_chars: 512,
    require_untrusted_envelope: true,
  },
  bindings: [
    { intent: "cursor_left", kind: "mouse", target: "move_x", value: -24 },
    { intent: "cursor_right", kind: "mouse", target: "move_x", value: 24 },
    { intent: "select", kind: "mouse", target: "click", value: "left" },
    { intent: "cancel", kind: "keyboard", target: "key", value: "Escape" },
    { intent: "agent_focus", kind: "agent", target: "open_task", value: null },
  ],
};

const seedReport = {
  status: "idle",
  passed: 0,
  failed: 0,
  results: [
    {
      name: "reference_pipeline",
      status: "waiting",
      severity: "info",
      message: "synthetic EEG pipeline",
      details: {},
    },
    {
      name: "non_finite_input_is_rejected",
      status: "waiting",
      severity: "high",
      message: "NaN / Inf guard",
      details: {},
    },
    {
      name: "agent_prompt_like_payload_is_blocked",
      status: "waiting",
      severity: "high",
      message: "agent payload guard",
      details: {},
    },
  ],
};

function App() {
  const [settings, setSettings] = useState(defaultSettings);
  const [report, setReport] = useState(seedReport);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selectedPanel, setSelectedPanel] = useState("signal");
  const [intentCommand, setIntentCommand] = useState({
    intent: "select",
    confidence: 0.92,
    payloadText: "",
  });
  const [routeResult, setRouteResult] = useState(null);

  const readiness = useMemo(() => {
    const issues = [];
    if (settings.signal.lowpass_hz <= settings.signal.highpass_hz) issues.push("band");
    if (settings.safety.min_confidence < 0.5) issues.push("confidence");
    if (settings.safety.require_human_arm && !settings.safety.human_armed) issues.push("unarmed");
    if (settings.safety.emergency_stop) issues.push("stopped");
    return issues;
  }, [settings]);

  async function runSelfTest() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/self-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ settings }),
      });
      if (!response.ok) throw new Error(`API ${response.status}`);
      setReport(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "self-test failed");
      setReport({
        ...seedReport,
        status: "offline",
        results: seedReport.results.map((item) => ({
          ...item,
          status: "waiting",
          message: `${item.message} (API offline)`,
        })),
      });
    } finally {
      setBusy(false);
    }
  }

  async function routeIntent() {
    setBusy(true);
    setError("");
    try {
      const payload = intentCommand.payloadText.trim() ? { text: intentCommand.payloadText.trim() } : {};
      const response = await fetch(`${API_BASE}/api/control/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          settings,
          command: {
            intent: intentCommand.intent,
            confidence: Number(intentCommand.confidence),
            payload,
          },
        }),
      });
      if (!response.ok) throw new Error(`API ${response.status}`);
      setRouteResult(await response.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "route failed");
      setRouteResult({ blocked: true, reason: "api_offline", metadata: { message: "API offline" } });
    } finally {
      setBusy(false);
    }
  }

  function resetSettings() {
    setSettings(defaultSettings);
    setReport(seedReport);
    setRouteResult(null);
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="rail" aria-label="NeuroCore navigation">
        <div className="brand">
          <div className="brand-mark">
            <Waves size={24} />
          </div>
          <div>
            <strong>NeuroCore</strong>
            <span>EEG Control Runtime</span>
          </div>
        </div>
        <nav className="nav-list">
          <NavButton icon={<SlidersHorizontal />} label="Signal" active={selectedPanel === "signal"} onClick={() => setSelectedPanel("signal")} />
          <NavButton icon={<ShieldCheck />} label="Safety" active={selectedPanel === "safety"} onClick={() => setSelectedPanel("safety")} />
          <NavButton icon={<MousePointer2 />} label="Route" active={selectedPanel === "route"} onClick={() => setSelectedPanel("route")} />
          <NavButton icon={<Bot />} label="Agent" active={selectedPanel === "agent"} onClick={() => setSelectedPanel("agent")} />
          <NavButton icon={<Activity />} label="Self-test" active={selectedPanel === "test"} onClick={() => setSelectedPanel("test")} />
        </nav>
        <div className={`runtime-state ${readiness.length === 0 ? "ok" : "warn"}`}>
          {readiness.length === 0 ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          <span>{readiness.length === 0 ? "Ready" : "Review"}</span>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>EEG Runtime Settings</h1>
            <p>Decoder weights stay external. NeuroCore validates signal flow and envelopes decoded intents.</p>
          </div>
          <div className="topbar-actions">
            <button className="ghost-button" type="button" onClick={resetSettings}>
              <RotateCcw size={16} />
              Reset
            </button>
            <button className="primary-button" type="button" onClick={runSelfTest} disabled={busy}>
              <Play size={16} />
              {busy ? "Running" : "Run Self-test"}
            </button>
          </div>
        </header>

        <section className="status-grid">
          <Metric icon={<Cpu />} label="Device" value={settings.device.name} detail={`${settings.device.sampling_rate} Hz`} />
          <Metric icon={<Waves />} label="Channels" value={settings.device.channels.length} detail={settings.device.channels.join(", ")} />
          <Metric icon={<Gauge />} label="Threshold" value={settings.safety.min_confidence.toFixed(2)} detail="intent confidence" />
          <Metric
            icon={<ShieldCheck />}
            label="Arm"
            value={settings.safety.human_armed ? "Armed" : "Unarmed"}
            detail={settings.safety.require_human_arm ? "trusted state" : "not required"}
          />
          <Metric icon={<CircleStop />} label="Stop" value={settings.safety.emergency_stop ? "Armed" : "Clear"} detail="emergency state" />
        </section>

        <section className="layout">
          <div className="panel settings-panel">
            <PanelHeader icon={<Settings2 />} title={panelTitle(selectedPanel)} />
            {selectedPanel === "signal" && <SignalPanel settings={settings} setSettings={setSettings} />}
            {selectedPanel === "safety" && <SafetyPanel settings={settings} setSettings={setSettings} />}
            {selectedPanel === "route" && (
              <RoutePanel
                intentCommand={intentCommand}
                setIntentCommand={setIntentCommand}
                routeIntent={routeIntent}
                routeResult={routeResult}
                busy={busy}
                error={error}
              />
            )}
            {selectedPanel === "agent" && <AgentPanel settings={settings} setSettings={setSettings} />}
            {selectedPanel === "test" && <SelfTestPanel report={report} error={error} />}
          </div>

          <div className="panel route-panel">
            <PanelHeader icon={<MousePointer2 />} title="Intent Bindings" />
            <div className="binding-list">
              {settings.bindings.map((binding) => (
                <div className="binding-row" key={`${binding.intent}-${binding.kind}-${binding.target}`}>
                  <div className="binding-icon">{bindingIcon(binding.kind)}</div>
                  <div>
                    <strong>{binding.intent}</strong>
                    <span>{binding.kind} / {binding.target}</span>
                  </div>
                  <code>{String(binding.value ?? "envelope")}</code>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function NavButton({ icon, label, active, onClick }) {
  return (
    <button className={`nav-button ${active ? "active" : ""}`} type="button" onClick={onClick}>
      {React.cloneElement(icon, { size: 18 })}
      <span>{label}</span>
    </button>
  );
}

function Metric({ icon, label, value, detail }) {
  return (
    <div className="metric">
      <div className="metric-icon">{React.cloneElement(icon, { size: 19 })}</div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
        <small>{detail}</small>
      </div>
    </div>
  );
}

function PanelHeader({ icon, title }) {
  return (
    <div className="panel-header">
      <div>
        {React.cloneElement(icon, { size: 18 })}
        <h2>{title}</h2>
      </div>
    </div>
  );
}

function SignalPanel({ settings, setSettings }) {
  return (
    <div className="form-grid">
      <Field label="Device name">
        <input value={settings.device.name} onChange={(event) => update(settings, setSettings, "device.name", event.target.value)} />
      </Field>
      <Field label="Sampling rate">
        <NumberInput value={settings.device.sampling_rate} onChange={(value) => update(settings, setSettings, "device.sampling_rate", value)} />
      </Field>
      <Field label="Channels">
        <input
          value={settings.device.channels.join(", ")}
          onChange={(event) =>
            update(
              settings,
              setSettings,
              "device.channels",
              event.target.value.split(",").map((item) => item.trim()).filter(Boolean),
            )
          }
        />
      </Field>
      <Field label="Target Hz">
        <NumberInput value={settings.signal.target_sampling_rate} onChange={(value) => update(settings, setSettings, "signal.target_sampling_rate", value)} />
      </Field>
      <Field label="Highpass">
        <NumberInput value={settings.signal.highpass_hz} onChange={(value) => update(settings, setSettings, "signal.highpass_hz", value)} />
      </Field>
      <Field label="Lowpass">
        <NumberInput value={settings.signal.lowpass_hz} onChange={(value) => update(settings, setSettings, "signal.lowpass_hz", value)} />
      </Field>
      <Field label="Window seconds">
        <NumberInput value={settings.signal.window_seconds} step="0.05" onChange={(value) => update(settings, setSettings, "signal.window_seconds", value)} />
      </Field>
      <Field label="Step seconds">
        <NumberInput value={settings.signal.step_seconds} step="0.05" onChange={(value) => update(settings, setSettings, "signal.step_seconds", value)} />
      </Field>
    </div>
  );
}

function SafetyPanel({ settings, setSettings }) {
  return (
    <div className="form-grid">
      <Field label="Min confidence">
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={settings.safety.min_confidence}
          onChange={(event) => update(settings, setSettings, "safety.min_confidence", Number(event.target.value))}
        />
        <span className="range-value">{settings.safety.min_confidence.toFixed(2)}</span>
      </Field>
      <Field label="Max actions / sec">
        <NumberInput value={settings.safety.max_actions_per_second} onChange={(value) => update(settings, setSettings, "safety.max_actions_per_second", value)} />
      </Field>
      <Toggle
        label="Require human arm"
        checked={settings.safety.require_human_arm}
        onChange={(checked) => update(settings, setSettings, "safety.require_human_arm", checked)}
      />
      <Toggle
        label="Human armed"
        checked={settings.safety.human_armed}
        onChange={(checked) => update(settings, setSettings, "safety.human_armed", checked)}
      />
      <Toggle
        label="Emergency stop"
        checked={settings.safety.emergency_stop}
        onChange={(checked) => update(settings, setSettings, "safety.emergency_stop", checked)}
      />
      <Toggle
        label="Block prompt-like payloads"
        checked={settings.safety.block_prompt_like_agent_payloads}
        onChange={(checked) => update(settings, setSettings, "safety.block_prompt_like_agent_payloads", checked)}
      />
    </div>
  );
}

function AgentPanel({ settings, setSettings }) {
  return (
    <div className="form-grid">
      <Field label="Allowed tools">
        <textarea
          rows="4"
          value={settings.agent.allowed_tools.join("\n")}
          onChange={(event) =>
            update(
              settings,
              setSettings,
              "agent.allowed_tools",
              event.target.value.split("\n").map((item) => item.trim()).filter(Boolean),
            )
          }
        />
      </Field>
      <Field label="Payload chars">
        <NumberInput value={settings.agent.max_payload_chars} onChange={(value) => update(settings, setSettings, "agent.max_payload_chars", value)} />
      </Field>
      <Toggle
        label="Untrusted envelope"
        checked={settings.agent.require_untrusted_envelope}
        onChange={(checked) => update(settings, setSettings, "agent.require_untrusted_envelope", checked)}
      />
    </div>
  );
}

function RoutePanel({ intentCommand, setIntentCommand, routeIntent, routeResult, busy, error }) {
  return (
    <div className="route-simulator">
      <div className="form-grid route-form">
        <Field label="Intent">
          <input
            value={intentCommand.intent}
            onChange={(event) => setIntentCommand({ ...intentCommand, intent: event.target.value })}
          />
        </Field>
        <Field label="Confidence">
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={intentCommand.confidence}
            onChange={(event) => setIntentCommand({ ...intentCommand, confidence: Number(event.target.value) })}
          />
          <span className="range-value">{Number(intentCommand.confidence).toFixed(2)}</span>
        </Field>
        <Field label="Agent payload">
          <textarea
            rows="4"
            value={intentCommand.payloadText}
            onChange={(event) => setIntentCommand({ ...intentCommand, payloadText: event.target.value })}
            placeholder="Optional decoded text"
          />
        </Field>
      </div>
      <div className="route-actions">
        <button className="primary-button" type="button" onClick={routeIntent} disabled={busy}>
          <Play size={16} />
          Route Intent
        </button>
        {error && <span className="route-error">{error}</span>}
      </div>
      <pre className={`route-result ${routeResult?.blocked ? "blocked" : "allowed"}`}>
        {routeResult ? JSON.stringify(routeResult, null, 2) : "No route result yet"}
      </pre>
    </div>
  );
}

function SelfTestPanel({ report, error }) {
  return (
    <div className="test-panel">
      <div className={`summary-strip ${report.status}`}>
        <strong>{report.status}</strong>
        <span>{report.passed} passed / {report.failed} failed</span>
      </div>
      {error && <div className="error-line">{error}</div>}
      <div className="test-list">
        {report.results.map((result) => (
          <div className={`test-row ${result.status}`} key={result.name}>
            {result.status === "passed" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
            <div>
              <strong>{result.name}</strong>
              <span>{result.message}</span>
            </div>
            <small>{result.severity}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle-row">
      <span>{label}</span>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
    </label>
  );
}

function NumberInput({ value, onChange, step = "1" }) {
  return <input type="number" step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} />;
}

function update(settings, setSettings, path, value) {
  const [group, key] = path.split(".");
  setSettings({
    ...settings,
    [group]: {
      ...settings[group],
      [key]: value,
    },
  });
}

function panelTitle(panel) {
  return {
    signal: "Signal Plan",
    safety: "Safety Policy",
    route: "Intent Simulator",
    agent: "Agent Envelope",
    test: "Self-test",
  }[panel];
}

function bindingIcon(kind) {
  if (kind === "mouse") return <MousePointer2 size={17} />;
  if (kind === "keyboard") return <Keyboard size={17} />;
  return <Bot size={17} />;
}

createRoot(document.getElementById("root")).render(<App />);
