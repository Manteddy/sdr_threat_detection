# Dependencies

## Core (required)

| Package | Min version | What it's used for |
|---------|-------------|-------------------|
| `numpy` | 1.20.0 | All hot-path signal processing; vectorised FFT, PSD, CFAR |
| `pyqtgraph` | 0.13.0 | Live spectrum + waterfall plot widgets |
| `PyQt5` | 5.15.0 | GUI framework, `QApplication`, worker thread, signals |
| `pyyaml` | any | Loading `engine_config.yaml` into `EngineConfig` dataclasses |

## Hardware-only (Linux + PlutoSDR)

| Package | Min version | What it's used for |
|---------|-------------|-------------------|
| `pyadi-iio` | 0.0.20 | PlutoSDR / AD9361 interface (`PyAdiIQSource`) — **not installed on Mac** |

System dependency: `libiio0` + `libiio-utils` + `python3-libiio` via `apt`.

## Optional

| Package | Notes |
|---------|-------|
| `numba` | JIT-accelerates CFAR detection. Commented out in `requirements.txt`; install manually if needed. |
| `websockets` | Required only for `proximity_alert/ws_server.py` WebSocket bridge. Skip if not exposing alerts externally. |

## Install (Mac / simulator only)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyqtgraph PyQt5 pyyaml
```

## Install (Linux / full hardware)

```bash
sudo apt install -y libiio0 libiio-utils python3-libiio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pyyaml
pip install numba        # optional
```

## Notes

- `pyadi-iio` is intentionally absent from the Mac venv. `libiio` is awkward on macOS and the simulator does not need it.
- macOS Docker Desktop cannot map a PlutoSDR over USB — hardware deployment is Linux-only.
- `run.sh` calls `.venv/bin/python` directly so no `source .venv/bin/activate` is needed per session.
