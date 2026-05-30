# Implementation Plan — One `capture()` for hardware + simulator

> Approved decisions (this is the source of truth for what gets built):
>
> 1. **Lift Pluto buffer cap** so both sources can deliver `num_frames × fft_size` samples — closes the 4× sync-buffer divergence flagged in the Review 2 analysis.
> 2. **Anti-alias filter inside `SignalReader`** at `bandwidth_hz` for every capture — eliminates edge-of-window aliasing in sim, no-op on real hardware (already has analog LPF).
> 3. **DC spike stays source-side** — real Pluto leakage is intrinsic; the simulator keeps reproducing it.
> 4. **Dual-channel becomes a `SignalReader` concern** — sources emit one channel only; `dir_iq = omni_iq.copy()` always. We are not using two antennas.
> 5. **Rename `SDRBackend` → `SignalReader`** as the public name visible to the engine.

## 1. Context

Two `capture()` implementations (real and simulated) produce data that the engine treats identically, but their internal rules diverge in ways that bite — most painfully the 4× sample-count difference that makes OS-CFAR sync detection work in sim and silently fail on hardware. The architectural intent has always been **"signal is signal"**: the engine never knows where IQ came from. This refactor enforces that contract.

## 2. Shape after the refactor

```
SpectrumEngine.attach_backend(reader)
                 │
                 ▼
   ┌────────────────────────────────────────┐
   │  SignalReader                          │   (single capture function)
   │  ───────────────────────────────────   │
   │  capture(center, bw, dwell, frames):   │
   │    1. source.tune(center)              │
   │    2. sleep pll_settle                 │
   │    3. raw = source.acquire(n_req)      │
   │    4. raw /= adc_full_scale            │
   │    5. raw = bandlimit(raw, fs, bw)     │
   │    6. omni = reshape (frames, fft)     │
   │    7. dir = omni.copy()                │   (single-antenna mirror)
   │    8. sleep dwell budget if realtime   │
   │    9. return IQCapture                 │
   └──────────────┬─────────────────────────┘
                  │  IQSource interface
       ┌──────────┴──────────┐
       ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│ PyAdiIQSource    │  │ SimulatedIQSource│
│ ──────────────── │  │ ──────────────── │
│ tune(): destroy  │  │ tune(): store    │
│   buffer + rx_lo │  │   current LO     │
│ acquire(n):      │  │ acquire(n):      │
│   rx() → ≥ n raw │  │   synth scene at │
│   ADC samples    │  │   LO + bw, scale │
│                  │  │   to ADC units   │
└──────────────────┘  └──────────────────┘
```

## 3. New / changed files

| Path | Status | Purpose |
|------|--------|---------|
| `spectrum_engine/iq_source.py` | **new** | `IQSource` ABC + the `IQCapture` / `HardwareLimits` dataclasses (moved from `sdr_backend.py`). |
| `spectrum_engine/signal_reader.py` | **new** | `SignalReader.capture()` — the single capture function the engine sees. Owns normalisation, anti-alias filter, framing, dual mirror, dwell budget. |
| `spectrum_engine/sources/__init__.py` | **new** | Source package marker. |
| `spectrum_engine/sources/pyadi.py` | **new** | `PyAdiIQSource` — tune via `rx_lo`, acquire via `rx()`. Sets `rx_buffer_size = num_frames × fft_size` at init (lifting the historical 2048-sample cap). Single-channel: ignores Pluto's dual-RX detection. |
| `spectrum_engine/sim/iq_source.py` | **new** | `SimulatedIQSource` — owns `Scene`, current LO, current bandwidth. `acquire(n)` synthesises in-window emitters + noise + DC spike, scales output by `adc_full_scale` so `SignalReader`'s divide step normalises it correctly. |
| `spectrum_engine/__init__.py` | edit | Export `SignalReader`, `IQCapture`, `HardwareLimits`. |
| `spectrum_engine/sim/__init__.py` | edit | Export `SimulatedIQSource` (in addition to scenarios / scene). |
| `spectrum_engine/engine.py` | edit | `attach_backend(reader: SignalReader)` — same interface, new type hint. Internally just calls `reader.capture(...)`. |
| `signal_pipeline/base.py` + `classic.py` | edit | Update `IQCapture` import from new location. |
| `drone_detector_enhanced.py` | edit | `_connect_hardware` builds `SignalReader(PyAdiIQSource(...), cfg)`; `_connect_simulator` builds `SignalReader(SimulatedIQSource(...), cfg)`. Scene swap goes through `reader.source.set_scene(...)`. |
| `spectrum_engine/sdr_backend.py` | **delete** | Replaced by `iq_source.py` (dataclasses) + `sources/pyadi.py` (logic). |
| `spectrum_engine/sim/backend.py` | **delete** | Replaced by `sim/iq_source.py`. |
| `sdr_threat_detection_ProjectDefinition.md` | edit | §2 diagram, §3 map (new files, removed files), §10 footguns. |

## 4. Interface contracts

### `IQSource`
```python
class IQSource(ABC):
    def get_limits(self) -> HardwareLimits: ...
    def tune(self, center_hz: float) -> None: ...
    def acquire(self, n_samples: int) -> np.ndarray:
        """Returns complex64, raw ADC units (range ~ ±adc_full_scale).
        Length ≥ n_samples preferred; SignalReader handles short returns."""
```

Sources are **single-channel**. The dual-antenna pathway is no longer needed.

### `SignalReader`
```python
class SignalReader:
    def __init__(self, source: IQSource, cfg: EngineConfig, *, realtime: bool = True): ...
    def get_limits(self) -> HardwareLimits: ...
    @property
    def source(self) -> IQSource: ...   # GUI uses reader.source.set_scene(...)
    def capture(self, center_hz, bandwidth_hz, dwell_s, num_frames) -> IQCapture: ...
```

`realtime=False` is for tests (skips dwell sleep). Default `True` for production.

## 5. Behavioural changes that ship with the refactor

| Change | Effect |
|--------|--------|
| **`rx_buffer_size = num_frames × fft_size`** at `PyAdiIQSource.__init__` (default 8192 for fine scan) | Hardware delivers requested frames. Sync feature now resolvable on real Pluto. |
| Anti-alias filter to `bandwidth_hz` in `SignalReader` | Simulator no longer aliases edge-of-window emitters; hardware unchanged (analog LPF already does this). |
| Single-channel everywhere | `dir_iq = omni_iq.copy()` mirror, always. Dropping dual-channel deletes ~100 lines of detection + duplicate synth code. |
| Sim emits raw-ADC-scale IQ | Normalisation moved to `SignalReader.capture` step 4 — one code path, sim + hardware identical. |

## 6. Risk + mitigation

| Risk | Mitigation |
|------|------------|
| Lifting `rx_buffer_size` past 2048 may re-trigger the "interleaved layout" pyadi-iio bug ([sdr_backend.py:131](spectrum_engine/sdr_backend.py:131) comment). I can't test it without hardware. | `PyAdiIQSource` accepts a `rx_buffer_size_override` kwarg. Default uses cfg-derived size; user can flip to 2048 if the lifted size breaks captures on their Pluto. Document the fallback. |
| Code currently imports `IQCapture` from `spectrum_engine.sdr_backend`. | Keep a one-line re-export shim during the migration; remove after the final commit verifies all consumers updated. |
| GUI's scene swap goes through `set_scene` on `SimulatedSDRBackend` directly. | New path: `self._engine_backend` becomes the `SignalReader`. `set_scene` calls become `self._engine_backend.source.set_scene(...)`. Single-line change in `_on_scene_changed` + `_connect_simulator`. |

## 7. Implementation order (commits)

| Commit | What lands | Verifiable | Estimated time |
|--------|-----------|------------|----------------|
| **A** — Foundation | `iq_source.py` (ABC + dataclasses), `signal_reader.py`. Old `sdr_backend.py` re-exports the dataclasses for compat. Nothing else changes. | Tree imports clean; existing tests still pass. | 1.5 h |
| **B** — Sources | `sources/pyadi.py`, `sim/iq_source.py`. Old `SDRBackend` and `SimulatedSDRBackend` still exist, still work. | Headless: `SignalReader(SimulatedIQSource(...))` ↔ engine produces identical detections to old path. | 2 h |
| **C** — Migration | Engine + GUI switch to `SignalReader`. Delete `sdr_backend.py` (keep dataclasses in `iq_source.py`) and `sim/backend.py`. | Full GUI lifecycle headless: connect, swap scene, swap processor, detect on/off, disconnect — all green. OSCFAR classifier on all 7 presets. | 2 h |
| **D** — Lifted cap + anti-alias + docs | `rx_buffer_size_override` path. Anti-alias FFT mask in `SignalReader`. Doc updates. | Headless: `cap.omni_iq.shape == (16, 512)`. PSD outside `bandwidth_hz/2` ≈ noise floor. | 1.5 h |

**Total: ~7 hours**, four sequential commits, each independently revertible.

## 8. Verification per phase

After **A**: `python -c "from spectrum_engine import SpectrumEngine, IQCapture, HardwareLimits; from spectrum_engine.signal_reader import SignalReader"` succeeds.

After **B**: `SimulatedIQSource(cfg, fpv_at(5.8e9))` + `SignalReader` delivers a fine capture; engine step against it produces ACTIVE cell at 5.8 GHz, same as before. OSCFAR processor still finds `ANALOG_FPV`.

After **C**: offscreen Qt GUI runs through the full lifecycle (`Src→Simulator → Simulate → Detect ON → swap scene → swap processor → Detect OFF → Disconnect → Src→Hardware → button "Receiver On"`) without exception. All 7 presets classify correctly under OSCFAR.

After **D**: `cap = reader.capture(5.8e9, 20e6, 0.030, 16)` returns `omni_iq.shape == (16, 512)` (not (4, 512)). A capture tuned 5 MHz off-centre of an 8 MHz FPV emitter shows ≥ 30 dB attenuation outside the `bandwidth_hz/2` mask.

## 9. Out of scope (explicitly deferred)

- **USRP / HackRF source.** Pattern is now in place — adding a third source is ~120 lines in `sources/`. Not part of this refactor.
- **Per-capture `bandwidth_hz` actually changing the SDR's RF filter.** Pyadi sets RF BW once at init. The argument is consumed by the anti-alias filter only.
- **Reviving dual-channel later.** If we ever want two antennas, the change is bound to `SignalReader.capture` step 7 (replace mirror with `source.acquire_pair`). The interface is forward-compatible.
- **`gen_analog_fpv` realism fixes** (chroma / audio sub-carriers, VSB asymmetry, pre-emphasis) from Review 1. Independent scope. Not part of this refactor.
