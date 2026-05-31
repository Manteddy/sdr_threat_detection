# Hardware Notes

Covers PlutoSDR-specific constraints that affect code decisions. See [footguns.md](footguns.md) for the full list of load-bearing "don't do this" rules.

## Target hardware

- PlutoSDR (ADALM-PLUTO) and AD9361-based clones.
- Stock Pluto tunes **325 MHz – 3.8 GHz**. **5.8 GHz requires a Pluto+ (AD9363) or a hardware-modified unit.**
- Sample rate and RF BW are driven from `engine_config.yaml hardware:`. Defaults: **40 MS/s, 40 MHz BW**; coarse FFT 512 bins, fine FFT 4096 bins.

## Dual-channel (directional antenna)

- Dual-channel mode (omni + directional) requires **Pluto 2r2t firmware**.
- Current codebase is single-antenna: `SignalReader` mirrors `omni_iq → dir_iq` in the `IQCapture`. No code change needed to enable dual-RX — only the firmware and wiring differ.

## Platform constraints

- **Linux only for real hardware.** macOS Docker Desktop cannot map a Pluto over USB. Simulator and replay run anywhere (Mac, Linux, CI).
- System packages required on Linux: `libiio0 libiio-utils python3-libiio`.

## Buffer ordering in `PyAdiIQSource`

`rx_buffer_size` is sized **once at `__init__`** to `fft_size_base × max(fine_frames, coarse_frames)`. Never resize at runtime — per-capture resizing has broken libiio's "interleaved sample layout" in the past (see [footguns.md](footguns.md)).

`tune()` must **destroy the buffer first, then set `rx_lo`**. The reverse order causes errno-0 stalls and zero scans (see [footguns.md](footguns.md)).

## ADC scale

The AD9361 ADC is 12-bit, delivered by pyadi-iio as int16 I/Q (range ≈ ±2048). `engine_config.yaml hardware.adc_full_scale` is set to 2048.0. `SignalReader.capture()` normalises to dBFS by dividing by this value.
