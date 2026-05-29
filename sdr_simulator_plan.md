# Implementation Plan — SDR Simulator for Offline Testing

> Companion to: [sdr_threat_detection_ProjectDefinition.md §8](sdr_threat_detection_ProjectDefinition.md). Sibling plan: [signal_processing_integration_plan.md](signal_processing_integration_plan.md).

## 1. Context

The full system today only works with a physical PlutoSDR / AD9361 attached over USB or IP. Every change to [`SpectrumEngine`](spectrum_engine/engine.py), [`spectrum_engine/detectors.py`](spectrum_engine/detectors.py), [`drone_detector_enhanced.py`](drone_detector_enhanced.py), or the planned `signal_pipeline/` package requires real hardware to verify — even basic unit tests cannot run in CI, and the upcoming Signal Processing Pipeline (project definition §8) has no way to be exercised offline.

**Goal:** a synthetic SDR backend that is a drop-in replacement for [`SDRBackend`](spectrum_engine/sdr_backend.py) and produces realistic baseband IQ for FPV, jammer, telemetry, and noise scenes. With it:

- The whole engine loop runs headless on a laptop with no SDR plugged in.
- The Signal Processing Pipeline can be unit-tested per stage and end-to-end.
- The enhanced GUI can be demoed against deterministic scenarios.
- CI can run integration tests deterministically.

The simulator is **not** a high-fidelity RF propagation model — it is a **functional** model whose only contract is: *the IQ buffer it emits, when run through the same PSD + CFAR + classifier code path as a real capture, produces statistically the same detections as the real hardware would have for a matching scene.*

---

## 2. What "drop-in replacement" means

The engine attaches a backend via [`SpectrumEngine.attach_backend(backend)`](spectrum_engine/engine.py:184) and only calls two methods on it:

| Method | Used at |
|--------|---------|
| `get_limits() -> HardwareLimits` | engine.py:187 (sets hw_max_hz, marks UNSUPPORTED cells) |
| `capture(center_hz, bandwidth_hz, dwell_s, num_frames) -> IQCapture` | engine.py:257 (every step) |

The simulator only has to honour those two methods and the `HardwareLimits` / `IQCapture` dataclasses already defined in [`sdr_backend.py`](spectrum_engine/sdr_backend.py:27). That gives us **duck-typed compatibility** — no inheritance, no abstract base class needed.

(If a refactor to a `Protocol`/`ABC` is wanted later it is a clean follow-up; not on the critical path.)

---

## 3. Module layout

```
spectrum_engine/
    sim/
        __init__.py          # exports SimulatedSDRBackend, Scene, Emitter, EmitterClass, scenarios
        backend.py           # SimulatedSDRBackend: capture(), get_limits()
        scene.py             # Scene, Emitter, EmitterClass dataclasses + scene composition
        synth.py             # signal synthesis primitives (per-class IQ generators)
        scenarios.py         # named test scenarios: empty_band, fpv_5800, jammer_2400, multi_emitter, drone_approach
        rng.py               # seeded RandomState helpers — deterministic by default
```

The `sim/` subpackage is intentionally inside `spectrum_engine/` (next to `sdr_backend.py`) because it shares the `IQCapture` / `HardwareLimits` definitions and `HardwareCfg` config schema. Importing it does **not** require `pyadi-iio` — that keeps headless test runs lean.

---

## 4. Public API

### 4.1 Backend

```python
from spectrum_engine.sim import SimulatedSDRBackend, scenarios

backend = SimulatedSDRBackend(
    scene=scenarios.fpv_at_5800_mhz(),       # or any Scene
    cfg=engine_cfg,                          # reuses HardwareCfg for sample_rate, fft_size_base, etc.
    seed=42,                                 # reproducibility
    dual_channel=False,                      # mirrors omni → dir if False (same as real)
    add_dc_spike=True,                       # simulate PlutoSDR LO leakage
    noise_floor_dbfs=-78.0,                  # ambient noise floor matching observed real captures
)

engine = SpectrumEngine(engine_cfg)
engine.attach_backend(backend)               # works because of duck typing

while running:
    snapshot = engine.step(time.monotonic())
```

### 4.2 Scene model

```python
@dataclass(frozen=True)
class Emitter:
    center_hz: float
    bandwidth_hz: float
    power_dbfs: float            # peak power seen at the RX with current path loss
    cls: EmitterClass            # ANALOG_FPV | BARRAGE_JAMMER | TELEMETRY_LINK | NOISE_BURST
    extra: dict = field(default_factory=dict)
    # extra keys (per class):
    #   ANALOG_FPV     : sync_h_hz=15625, sync_v_hz=50, mod_index=0.7
    #   BARRAGE_JAMMER : (none — flat noise)
    #   TELEMETRY_LINK : hop_rate_hz=0 (static carrier) or >0 (FHSS)
    #   NOISE_BURST    : on_duration_s, off_duration_s

@dataclass
class Scene:
    emitters: list[Emitter] = field(default_factory=list)
    noise_floor_dbfs: float = -78.0
    # Time evolution — None means static
    update: Optional[Callable[[Scene, float], None]] = None
```

`Scene` is mutable so a scenario can shift emitter positions/powers over wall-clock time. Scenarios like `drone_approach()` set `update` to a callback that walks `power_dbfs` along a log-distance curve.

### 4.3 Built-in scenarios

| Name | Purpose |
|------|---------|
| `empty_band()` | Pure noise across the whole tunable range. Sanity floor. |
| `fpv_at_5800_mhz(distance_m=80)` | One `ANALOG_FPV` emitter at 5.800 GHz with realistic 8 MHz BW + sync pulses. |
| `jammer_at_2400_mhz(bw_hz=20e6)` | One `BARRAGE_JAMMER` covering 2.39 - 2.41 GHz. |
| `telemetry_at_915_mhz(hop_rate_hz=100)` | Narrowband 500 kHz FHSS-like emitter. |
| `multi_emitter()` | FPV @ 5.8 + jammer @ 2.4 + telemetry @ 915 simultaneously. Stress test for scheduler + classifier. |
| `drone_approach(start_m=300, end_m=20, duration_s=30)` | Time-varying scenario: power rises as the drone closes; tests `DistanceEstimator` round-trip and Kalman convergence. |

---

## 5. Signal synthesis (`synth.py`)

All generators produce **complex64** IQ at the engine's configured sample rate and return baseband samples (centred at 0 Hz). The backend handles frequency-shifting each emitter into the tuned window.

### 5.1 ANALOG_FPV

Closest to the real thing for classifier validation:
- Wideband FM carrier with bandwidth ~ 8 MHz.
- Modulating signal = synthetic video luma envelope: sawtooth at the **horizontal sync rate (15.625 kHz)** modulated by a slow **vertical sync rate (50 Hz / 60 Hz)** square wave.
- FM-modulate that envelope onto the complex carrier with a configurable modulation index.
- Result has the canonical sinc-like FM spectrum **plus** strong cyclic features at 15.625 kHz in the envelope-power FFT — exactly what the Signal Processing Pipeline (Stage 3 sync detection) is looking for.

```python
def gen_analog_fpv(n_samples, fs, bandwidth_hz, sync_h_hz=15625,
                   sync_v_hz=50, mod_index=0.7, rng=None) -> np.ndarray
```

### 5.2 BARRAGE_JAMMER

Complex AWGN band-limited to `bandwidth_hz`:
- Generate white complex Gaussian, FFT-shift, mask outside `[-bw/2, +bw/2]`, IFFT.
- Power-normalize to the configured `power_dbfs`.
- High spectral flatness, zero cyclic structure → the classifier should label this `BARRAGE_JAMMER`.

```python
def gen_barrage_jammer(n_samples, fs, bandwidth_hz, rng=None) -> np.ndarray
```

### 5.3 TELEMETRY_LINK

Narrowband GFSK-like carrier (or pure tone for the static case):
- BW < 1 MHz.
- Optional `hop_rate_hz` > 0 for FHSS: jump to a new offset within the original BW every `1/hop_rate_hz` seconds.
- Classifier rule keys only on bandwidth, so even a static narrowband tone is enough for the v1 test.

```python
def gen_telemetry(n_samples, fs, bandwidth_hz, hop_rate_hz=0.0, rng=None) -> np.ndarray
```

### 5.4 Ambient noise

```python
def gen_awgn(n_samples, power_dbfs, rng=None) -> np.ndarray
```

Pure complex AWGN at the configured noise floor — added to every capture regardless of emitter content.

### 5.5 DC spike (optional)

If `add_dc_spike` is true, add a small DC offset (a few percent of full scale) to every capture so the engine's existing DC excision (`engine.py:300-303`) is exercised by tests. Disabling reveals whether engine logic depends on that excision.

---

## 6. Capture composition (`backend.py`)

`SimulatedSDRBackend.capture(center_hz, bandwidth_hz, dwell_s, num_frames)`:

1. **Resolve scene state.** If `scene.update` is set, call `scene.update(scene, monotonic_now - self._t0)` first to walk time-varying emitters.
2. **Allocate output**: `n_samples = self._cfg.fft_size_base * num_frames` complex64 zeros.
3. **For each emitter** in `scene.emitters` whose frequency overlaps `[center_hz - bw/2, center_hz + bw/2]`:
   - Synthesize baseband IQ via `synth.gen_*(...)` at the configured `sample_rate_hz`, length `n_samples`, BW = emitter BW.
   - Frequency-shift it by `delta = emitter.center_hz - center_hz` using `np.exp(2j*pi*delta*t)`.
   - Scale to `emitter.power_dbfs` (in linear amplitude relative to `adc_full_scale`).
   - Add to the output buffer.
4. **Add ambient noise** at `scene.noise_floor_dbfs`.
5. **Add DC spike** if enabled.
6. **Normalise by adc_full_scale** (matches `_read_natural` doing the same on real captures so PSD scaling matches).
7. **Reshape** to `(num_frames, fft_size_base)` for both `omni_iq` and `dir_iq` (same array twice when `dual_channel=False`). For `dual_channel=True`, add small uncorrelated noise on the dir channel to simulate independent paths.
8. **Sleep** for `pll_settle_s + max(dwell_s - n_samples/fs, 0)` to simulate retune + dwell time. Optional via `realtime=False` constructor flag for tests that want to run as fast as possible.
9. Return `IQCapture(center_hz, bandwidth_hz, timestamp=time.monotonic(), omni_iq, dir_iq)`.

**Performance:** at default config (`fft_size_base=512`, `num_frames=16`) the slowest scene (`multi_emitter`) needs ~3 FFT-shifts + 3 synth calls per capture. Budget < 5 ms so the engine's step rate is bounded by the simulator's `sleep`, not its CPU.

---

## 7. Wiring into the GUI and CLI

### 7.1 CLI flag

Add `--simulator <scenario>` to `drone_detector_enhanced.py`:

```bash
python drone_detector_enhanced.py --simulator fpv_5800
python drone_detector_enhanced.py --simulator drone_approach
```

When the flag is present, `_connect()` skips the pyadi-iio attempt and instantiates `SimulatedSDRBackend` + a stub `self.sdr` placeholder (or a thin shim object that satisfies the few `SweepWorker` accesses outside the engine path). The classic sweep loop should be disabled in simulator mode — engine loop only — to avoid a second consumer of the fake SDR.

### 7.2 GUI toggle (optional, follow-up)

A combo box "Source: [Hardware | Simulator: fpv_5800 | Simulator: jammer_2400 | ...]" next to the Connect button, replacing the CLI flag when shipped. Not on the critical path; the CLI is enough to unblock testing.

### 7.3 Headless mode

```bash
python -m spectrum_engine.sim --scenario fpv_5800 --duration 30 --log spectrum_logs/sim_fpv.jsonl
```

A small `__main__.py` in `spectrum_engine.sim` constructs the engine + simulator, runs N seconds of `engine.step()`, and dumps every snapshot's classifications + tracks to a JSONL file. This is the workhorse for unit / integration tests and for offline tuning of classifier thresholds.

---

## 8. Tests this enables

| Test | Scenario | Assertion |
|------|----------|-----------|
| `test_empty_band_no_tracks` | `empty_band()` | After 50 steps, `track_mgr.active_tracks == []` and median cell occupancy < 0.1. |
| `test_fpv_creates_track` | `fpv_at_5800_mhz()` | A track centred within ±2 MHz of 5.800 GHz appears within 5 fine scans. |
| `test_fpv_classified` | same | After integration with `signal_pipeline/`, a `ClassificationResult.classification == ANALOG_FPV` appears with confidence > 0.6. |
| `test_jammer_classified` | `jammer_at_2400_mhz()` | Classification `BARRAGE_JAMMER`; `AlertEngine` does **not** escalate to CRITICAL. |
| `test_drone_approach_distance` | `drone_approach()` | Kalman `DistanceEstimator.x` drops from ~300 m to ~30 m monotonically over the run; `inside_boundary` flips from False to True. |
| `test_multi_emitter_scheduler_fairness` | `multi_emitter()` | Each emitter's cell is revisited within its `max_revisit_interval_s` ≥ 95% of the time. |
| `test_pipeline_budget` | `fpv_at_5800_mhz()` | p95 of `SignalPipeline.run` < 8 ms over 1000 iterations. |

Tests live in a new `tests/` directory at the repo root. Use `pytest`. CI runs them on every push.

---

## 9. Implementation order

1. **Bare backend** — `SimulatedSDRBackend` that ignores the scene and returns pure AWGN. Confirm `SpectrumEngine.attach_backend(simulator)` works and `engine.step()` produces empty snapshots without crashing.
2. **`gen_awgn` + ambient noise** in capture composition. Confirm noise-floor estimate inside the engine matches the configured `noise_floor_dbfs` ± 1 dB.
3. **`gen_telemetry`** (simplest) + `telemetry_at_915_mhz()` scenario. Confirm the engine produces a `DetectedRegion` around 915 MHz.
4. **`gen_barrage_jammer`** + `jammer_at_2400_mhz()` scenario. Confirm wideband region detected.
5. **`gen_analog_fpv`** + `fpv_at_5800_mhz()` scenario. The deepest synthesis work — sync pulses and FM modulation must be right for the Signal Processing Pipeline's sync-detection stage to fire.
6. **Time-varying scene support** — `scene.update` callable + `drone_approach()` scenario.
7. **CLI flag** in `drone_detector_enhanced.py` + headless `__main__.py` in `spectrum_engine.sim`.
8. **`tests/`** — write the table above against the simulator. Run with and without `signal_pipeline/` to keep both layers tested.

---

## 10. Verification

- **Round-trip sanity**: feed the simulator's IQ through the engine's existing PSD path, log the resulting `noise_floor_db` and `peak_db` per cell, and confirm they match the configured `noise_floor_dbfs` / emitter `power_dbfs` within a small tolerance (±2 dB).
- **No real-hardware regressions**: with the simulator landed, the existing physical SDR path must still attach and run unchanged. Manual smoke run of `drone_detector_enhanced.py` against actual hardware.
- **Determinism**: same `seed` + same scenario + same engine config → byte-identical `EngineSnapshot.last_psd_db` across runs. Anything stochastic must be driven by the seeded RNG, never `time.time()` or unseeded `np.random`.
- **Performance**: simulator capture < 5 ms; engine `step()` < 15 ms (≥ 60 sweeps/sec) on a developer laptop without GPU.

---

## 11. Out of scope (deferred)

- Multipath / fading — single LOS path is enough for the classifier; multipath simulation is a future enhancement.
- Real RF impairments beyond the DC spike: IQ imbalance, phase noise, ADC quantization. Not blockers for the v1 simulator.
- IIO emulation — we are not building a `libiio`-compatible fake device; we replace `SDRBackend` directly. Easier, less code, and immune to libiio API drift.
- Scenario authoring UI — Python scenario constructors are enough until/unless non-engineers need to author tests.
