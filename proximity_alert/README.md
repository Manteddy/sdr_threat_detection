# proximity_alert

Embeddable drone proximity warning package.
Accepts signal updates and produces threat-level alerts with trend direction (rising / stable / falling) and color coding.

Designed for integration with mission-control dashboards, GCS (Ground Control Station) UIs, or any Python/web application.

---

## Quick Start (Demo)

No hardware needed:

```bash
cd pluto_drone_detector
python -m proximity_alert
```

A window appears with simulated drone signals approaching and retreating.

---

## Integration Options

### Option A: Embed PyQt5 Widget

Drop the panel into any PyQt5 layout:

```python
from proximity_alert.widget import ProximityAlertPanel

panel = ProximityAlertPanel()
your_layout.addWidget(panel)

# Each sweep / update cycle:
alert = panel.push(
    freq_ghz=5.800,
    signal_dbfs=-38.0,
    distance_m=87.0,
    confidence=0.65,
)
```

The panel shows:
- Per-frequency rows with signal value, distance, trend arrow, threat level
- Top banner warning for CRITICAL and APPROACHING threats
- Flashing red border when CRITICAL

### Option B: Use Engine Only (Headless)

```python
from proximity_alert import AlertEngine, ThreatLevel

engine = AlertEngine(enter_m=95, exit_m=120)

alert = engine.update(
    freq_ghz=5.800,
    signal_dbfs=-38.0,
    distance_m=87.0,
    confidence=0.65,
)

if alert.threat != ThreatLevel.NONE:
    print(alert.summary_line())
    # ⚠ CRITICAL | 5.800 GHz | -38 dBFS | ~87m | ▲ rising
```

### Option C: WebSocket Bridge (Web UIs)

For web-based dashboards (e.g. mission-control panels running in a browser):

```python
from proximity_alert import AlertEngine
from proximity_alert.ws_server import AlertWSServer
import asyncio

engine = AlertEngine()
server = AlertWSServer(engine, host="0.0.0.0", port=9800)

async def main():
    await server.start()
    while True:
        alert = engine.update(freq_ghz=5.8, signal_dbfs=-38, distance_m=87)
        await server.broadcast(alert)
        await asyncio.sleep(0.2)

asyncio.run(main())
```

Then in JavaScript / any web client:

```javascript
const ws = new WebSocket("ws://localhost:9800");
ws.onmessage = (e) => {
    const alert = JSON.parse(e.data);
    // alert.threat = "CRITICAL" | "APPROACHING" | "DETECTED" | "NONE"
    // alert.trend  = "RISING" | "STABLE" | "FALLING" | "UNKNOWN"
    // alert.color  = "#ff3b30" | "#ff9500" | "#ffd60a" | "#8e8e93"
    // alert.signal_dbfs, alert.distance_m, alert.freq_ghz, ...
    showWarning(alert);
};
```

---

## Threat Levels

| Level | Color | Meaning |
|-------|-------|---------|
| **CRITICAL** | Red `#ff3b30` | Drone inside ~100 m boundary, or approaching + signal rising |
| **APPROACHING** | Orange `#ff9500` | Drone detected within 250 m |
| **DETECTED** | Yellow `#ffd60a` | Signal above detection threshold |
| **NONE** | Gray `#8e8e93` | Below detection threshold |

## Trend Direction

| Trend | Arrow | Meaning |
|-------|-------|---------|
| **RISING** | ▲ | Signal getting stronger (drone likely approaching) |
| **STABLE** | ▬ | Signal steady |
| **FALLING** | ▼ | Signal getting weaker (drone likely moving away) |
| **UNKNOWN** | ? | Not enough history yet |

---

## JSON Alert Format (WebSocket)

```json
{
    "freq_ghz": 5.800,
    "signal_dbfs": -38.0,
    "distance_m": 87.0,
    "confidence": 0.65,
    "threat": "CRITICAL",
    "trend": "RISING",
    "color": "#ff3b30",
    "trend_arrow": "▲",
    "timestamp": 1709312400.123
}
```

---

## Connecting to the Main Detector

To feed real SDR data from `drone_detector_enhanced.py`:

```python
from proximity_alert import AlertEngine

engine = AlertEngine()

# Inside _on_sweep_done, after distance estimation:
for freq in self.monitor_freqs_ghz:
    de = self.distance_estimators.get(freq)
    hist = self.monitor_history.get(freq)
    if hist and hist["omni"]:
        alert = engine.update(
            freq_ghz=freq,
            signal_dbfs=hist["omni"][-1],
            distance_m=de.x if de else None,
            confidence=de.confidence if de else 0.0,
        )
```

---

## Requirements

- Python 3.8+
- PyQt5 (for widget and demo)
- websockets (only for WebSocket bridge, optional)

```bash
pip install PyQt5
pip install websockets  # optional, for ws_server
```

---

## Package Structure

```
proximity_alert/
    __init__.py       - Package entry, exports AlertEngine / ProximityAlert / TrendDirection
    engine.py         - Core: alert engine, trend detection, threat classification
    widget.py         - Embeddable PyQt5 ProximityAlertPanel
    ws_server.py      - WebSocket server for web UI integration
    demo.py           - Simulated demo (no hardware needed)
    __main__.py       - python -m proximity_alert entry point
    README.md         - This file
```
