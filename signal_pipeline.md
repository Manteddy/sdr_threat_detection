# Signal Pipeline: From Radio Wave to Spectrum Display and Alert

This document explains, in detail, how the detector turns an incoming radio signal into:

1. a spectrum plot,
2. a waterfall plot,
3. monitor-frequency history plots,
4. a signal alert,
5. and, in the enhanced version, a proximity/threat alert.

It is written for a first-year student who has basic physics and programming knowledge, but may not yet know software-defined radio (SDR), digital signal processing (DSP), FFTs, CFAR, or Kalman filters.

The document describes the current project behavior in:

- `drone_detector.py`
- `drone_detector_enhanced.py`
- `proximity_alert/engine.py`
- `proximity_alert/widget.py`

Important note: this file describes detection and display logic. It does not prove that a received signal is definitely a drone. The software detects signal energy near selected frequencies and labels it according to the configured detection rules.

---

## 1. Big Picture

The detector is trying to answer two questions:

1. "What radio energy is present across the selected frequency range?"
2. "Is there enough signal energy near one of my monitored frequencies to show an alert?"

The full path is:

```text
radio wave in air
-> antenna voltage/current
-> SDR analog receiver
-> ADC samples
-> complex IQ numbers
-> windowed time-domain frame
-> FFT frequency bins
-> dBFS spectrum
-> stitched wideband spectrum
-> spectrum plot and waterfall plot
-> local peak near monitored frequencies
-> threshold/CFAR decision
-> persistence counter
-> optional distance estimate
-> UI alert
```

Each arrow means the information has changed form. Sometimes it changes physical form, such as an electromagnetic wave becoming voltage at an antenna. Sometimes it changes mathematical form, such as time-domain samples becoming frequency-domain bins.

---

## 2. Important Terms Before the Pipeline

### 2.1 Signal

A signal is information carried by a changing physical quantity.

In this project, the physical signal starts as an electromagnetic radio wave. Later, inside the computer, the signal is represented as numbers.

The signal changes representation many times:

- electromagnetic field in air,
- tiny voltage/current at the antenna,
- analog baseband waveform inside the SDR,
- digital ADC samples,
- complex IQ samples,
- FFT bins,
- dBFS values,
- UI pixels and alert states.

### 2.2 Frequency

Frequency means "how many cycles per second".

The unit is hertz (Hz):

- 1 Hz = 1 cycle per second.
- 1 kHz = 1,000 Hz.
- 1 MHz = 1,000,000 Hz.
- 1 GHz = 1,000,000,000 Hz.

The detector defaults to looking from 5.6 GHz to 6.0 GHz. That is a 400 MHz span.

### 2.3 RF

RF means radio frequency.

An RF signal is an electromagnetic wave at radio frequencies. The detector receives RF energy through an antenna.

### 2.4 Antenna

An antenna converts between electromagnetic waves in space and electrical signals in conductors.

On receive:

- the radio wave reaches the antenna,
- the electric and magnetic fields push charges in the antenna conductor,
- that creates a tiny electrical voltage/current,
- the SDR receives that electrical signal.

The code supports two conceptual receive paths:

- RX1 / omni: a broad-coverage antenna path.
- RX2 / directional: a directional antenna path.

If the hardware does not provide two different receive channels, the software mirrors RX1 into RX2 so the UI still has two panels.

### 2.5 SDR

SDR means software-defined radio.

An SDR is radio hardware whose behavior is controlled mostly by software. The PlutoSDR / AD9361 hardware handles the analog RF reception and ADC conversion. Python code controls settings such as:

- center frequency,
- sample rate,
- bandwidth,
- hardware gain,
- enabled receive channels.

### 2.6 ADC

ADC means analog-to-digital converter.

It converts a continuous analog voltage into digital numbers sampled at regular time intervals.

Example:

```text
continuous analog wave
-> ADC
-> [104, 117, 93, -20, ...]
```

The detector later normalizes these raw ADC-like values using `ADC_FULL_SCALE = 2048.0`.

### 2.7 IQ Samples

IQ means in-phase and quadrature.

Instead of storing one real number per sample, an SDR commonly stores a complex number:

```text
I + jQ
```

Where:

- `I` is the real part.
- `Q` is the imaginary part.
- `j` is the imaginary unit.

Why IQ matters:

- A real-only signal cannot easily distinguish frequencies above and below the tuned center frequency.
- A complex IQ signal can represent both amplitude and phase.
- IQ samples are ideal for FFT-based spectrum analysis.

In Python/NumPy, the detector stores IQ samples as `np.complex64`.

### 2.8 Time Domain

Time domain means the signal is described as values over time.

An IQ array like this is time-domain data:

```text
sample 0:  0.10 + 0.02j
sample 1:  0.08 + 0.04j
sample 2: -0.01 + 0.07j
...
```

The array tells us "what the received waveform looked like at each time sample."

### 2.9 Frequency Domain

Frequency domain means the signal is described as energy at different frequencies.

After an FFT, we get bins like:

```text
bin 0: frequency A has -83 dBFS
bin 1: frequency B has -81 dBFS
bin 2: frequency C has -38 dBFS
...
```

The frequency domain is what the spectrum plot displays.

### 2.10 FFT

FFT means Fast Fourier Transform.

Fourier transform idea:

- any waveform can be described as a mixture of sine/cosine waves at different frequencies.
- the FFT is a fast algorithm that calculates that mixture from sampled data.

In this project:

```python
fft(iq_norm * WINDOW)
```

takes a short time-domain IQ frame and produces complex frequency bins.

### 2.11 Bin

An FFT bin is one slot in the FFT output.

If `FFT_SIZE = 512`, the FFT returns 512 bins. Each bin corresponds to a small frequency interval inside the currently tuned 20 MHz slice.

Approximate bin width:

```text
20 MHz / 512 = 39,062.5 Hz per bin
```

This means each bin represents about 39 kHz of frequency width.

### 2.12 dB and dBFS

dB is a logarithmic unit. It compresses large ranges into manageable numbers.

For amplitude ratios:

```text
dB = 20 * log10(amplitude_ratio)
```

dBFS means decibels relative to full scale.

In an ADC system:

- 0 dBFS means maximum representable digital full-scale amplitude.
- negative values mean below full scale.
- -40 dBFS is weaker than -20 dBFS.
- -100 dBFS is very weak.

The detector clips display values between:

```text
DB_MIN = -140
DB_MAX = 0
```

### 2.13 Noise Floor

Noise floor means the background signal level when no strong signal is present.

Sources include:

- thermal noise,
- receiver electronics,
- interference,
- ADC quantization,
- environment.

The detector estimates noise floor using median spectrum values and smooths them over time.

### 2.14 Threshold

A threshold is a line that decides whether something is "big enough".

Simple example:

```text
if measured_signal > threshold:
    alert
```

In the simple detector, the user threshold defaults to:

```text
DEFAULT_THRESHOLD_DBFS = -40
```

### 2.15 CFAR

CFAR means Constant False Alarm Rate.

It is a detection method that adapts the threshold to local noise.

Instead of using one fixed threshold everywhere, CFAR asks:

"Is this bin much stronger than the nearby bins around it?"

That helps when some parts of the spectrum have higher background noise than others.

### 2.16 Waterfall

A waterfall is a history of spectra over time.

One horizontal line is one sweep result. New lines are appended over time, creating an image:

```text
frequency -> left to right
time      -> one line after another
color     -> signal strength
```

### 2.17 Alert

An alert is a UI state produced when signal energy near a monitored frequency passes the software rules.

The alert is not directly created by the antenna. It is created after:

- FFT conversion,
- peak extraction,
- threshold or CFAR comparison,
- counter persistence,
- optional distance/threat processing.

---

## 3. Constants Used by the Detector

The detector uses these important constants.

### 3.1 Sweep range

Default start:

```text
DEFAULT_START_GHZ = 5.6
```

Default end:

```text
DEFAULT_END_GHZ = 6.0
```

Meaning:

- start at 5.6 GHz,
- stop at 6.0 GHz,
- cover 400 MHz total.

### 3.2 Sample rate

```text
SAMPLE_RATE = 20,000,000 samples/second
```

This means the SDR collects 20 million complex samples per second.

### 3.3 RF bandwidth

```text
BANDWIDTH_HZ = 20,000,000 Hz
```

This configures the receiver bandwidth near the tuned center frequency.

### 3.4 FFT size

```text
FFT_SIZE = 512
```

Each FFT uses 512 time-domain IQ samples and returns 512 frequency bins.

### 3.5 Sweep step

```text
SWEEP_STEP_HZ = 20,000,000 Hz
```

The software retunes the SDR in 20 MHz steps.

Example for a sweep from 5.6 to 6.0 GHz:

```text
5.600 GHz
5.620 GHz
5.640 GHz
...
5.980 GHz
```

Each step produces one 512-bin spectrum slice.

### 3.6 PLL settle time

```text
PLL_SETTLE_S = 0.0008 seconds
```

PLL means phase-locked loop. It is the hardware circuit that creates the local oscillator frequency.

After retuning, the code waits 0.8 ms so the receiver has time to stabilize before reading samples.

### 3.7 ADC full scale

```text
ADC_FULL_SCALE = 2048.0
```

The code divides raw IQ values by this number.

This transforms raw device-scale numbers into normalized values where "1.0" is approximately full scale.

### 3.8 Display limits

```text
DB_MIN = -140.0
DB_MAX = 0.0
DYNAMIC_RANGE = 60.0
```

Meaning:

- anything below -140 dBFS is displayed as -140,
- anything above 0 dBFS is displayed as 0,
- waterfall brightness maps about 60 dB above the noise floor into the visible color range.

---

## 4. Physical Signal Arrival

### 4.1 Radio wave in the air

An FPV video transmitter or other RF source radiates energy. At 5.8 GHz, the wavelength is about:

```text
wavelength = speed_of_light / frequency
           = 300,000,000 m/s / 5,800,000,000 Hz
           = about 0.052 m
           = about 5.2 cm
```

This short wavelength is why antennas for this band can be physically small.

### 4.2 Propagation effects

Before the signal reaches the antenna, it can change due to:

- distance path loss: signal gets weaker with distance,
- multipath: reflections from surfaces arrive with different delays,
- fading: signals can add or cancel depending on phase,
- polarization mismatch: antenna orientation affects received strength,
- obstruction: objects absorb or reflect energy,
- interference: other transmitters add energy in the same band.

At this stage, the signal is still physical electromagnetic energy.

### 4.3 Antenna conversion

The antenna converts the electromagnetic wave into an electrical signal.

Information form changes:

```text
electromagnetic field
-> voltage/current at antenna terminals
```

The frequency content is still present, but now it is an electrical waveform instead of a wave traveling through space.

---

## 5. SDR Analog Receiver Stage

The SDR receives the antenna signal and prepares it for digitization.

### 5.1 Tuning with local oscillator

The software sets:

```python
sdr.rx_lo = current_freq
```

`rx_lo` means receive local oscillator.

The local oscillator is an internal signal generated by the SDR. It is used to shift the desired RF frequency range down to baseband.

Baseband means centered near 0 Hz after mixing. This makes the signal easy to sample and process digitally.

### 5.2 Mixing

Mixing combines the incoming RF with the local oscillator.

Conceptually:

```text
incoming RF around 5.8 GHz
mixed with LO around 5.8 GHz
-> baseband signal around 0 Hz
```

The frequency has not disappeared. It has been shifted into a lower-frequency representation.

### 5.3 Analog filtering

The SDR applies analog filtering around the selected bandwidth.

The code configures:

```python
sdr.rx_rf_bandwidth = 20_000_000
```

This means the receiver tries to keep a 20 MHz-wide slice and reduce energy outside that slice.

### 5.4 Hardware gain

The detector sets manual gain:

```python
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = gain_slider_value
```

If RX2 is available:

```python
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan1 = gain_slider_value
```

Gain is real signal amplification before or inside the receiver chain.

Important:

- Increasing gain can make weak signals easier to see.
- Too much gain can overload/saturate the receiver.
- Saturation creates misleading spectrum artifacts.

This is the main true signal "boost" before software processing.

---

## 6. ADC and IQ Creation

### 6.1 Sampling

The ADC samples the baseband signal at:

```text
20,000,000 samples per second
```

Sampling converts continuous time into discrete sample points.

### 6.2 Complex sample format

The SDR returns complex IQ samples.

Each sample has:

```text
I = real component
Q = imaginary component
```

The Python representation is:

```python
np.complex64
```

This means:

- 32-bit floating real part,
- 32-bit floating imaginary part,
- total 64 bits per complex sample.

### 6.3 What `sdr.rx()` returns

The code reads:

```python
raw = self.sdr.rx()
```

Depending on hardware and driver, `raw` can be:

- a list/tuple of channel arrays,
- a 2D NumPy array,
- a single channel array.

The code normalizes these cases into:

```python
omni_iq
dir_iq
```

These are the two arrays used for later processing.

---

## 7. Software Sweep Loop

The detector does not process the whole 5.6 to 6.0 GHz range at once. The SDR only receives a 20 MHz slice at a time.

So the software sweeps.

### 7.1 Sweep meaning

A sweep is a repeated sequence:

```text
tune to frequency 1
read IQ
FFT
store bins

tune to frequency 2
read IQ
FFT
store bins

...

when all steps are complete:
emit one full wideband sweep result
```

### 7.2 Per-step operations

For each column/step `col`, the code does:

```python
self.sdr.rx_destroy_buffer()
self.sdr.rx_lo = int(self.current_freq)
time.sleep(PLL_SETTLE_S)
raw = self.sdr.rx()
```

Explanation:

- `rx_destroy_buffer()` clears previous buffered samples.
- `rx_lo` retunes the receiver.
- `sleep` allows hardware to settle.
- `rx()` captures new samples.

### 7.3 Current frequency update

After each step:

```python
self.current_freq += SWEEP_STEP_HZ
```

So every step moves 20 MHz higher.

---

## 8. Channel Handling

### 8.1 Dual channel case

If hardware supports two real receive channels, the code extracts:

```python
omni_iq = channel 0
dir_iq = channel 1
```

The UI labels these as:

- RX1 / omni,
- RX2 / directional.

### 8.2 Single channel fallback

If two real channels are not available:

```python
dir_iq = omni_iq.copy()
```

This means RX2 is not an independent measurement. It is a mirrored display of RX1.

The information does not improve in this case. It is duplicated so the UI layout can remain consistent.

### 8.3 Data shape after channel handling

After this stage, the software expects:

```text
omni_iq: 1D complex array
dir_iq:  1D complex array
```

Each element is one complex IQ sample.

---

## 9. Frame Length Normalization

The FFT needs a fixed number of samples.

The code uses:

```text
FFT_SIZE = 512
```

### 9.1 If too few samples arrive

If the received array is shorter than 512 samples:

```python
np.pad(..., (0, FFT_SIZE - n))
```

This adds zeros at the end.

Zero padding here is mainly a safety step so the FFT always has enough input samples.

### 9.2 If more samples arrive

The main display FFT uses:

```python
iq[:FFT_SIZE]
```

So it uses the first 512 samples for the single-FFT display path.

### 9.3 Information change

Before:

```text
variable length IQ array
```

After:

```text
exactly 512 complex samples
```

This fixed size makes each frequency step comparable.

---

## 10. ADC Full-Scale Normalization

The code does:

```python
omni_norm = omni_iq[:FFT_SIZE] / ADC_FULL_SCALE
dir_norm = dir_iq[:FFT_SIZE] / ADC_FULL_SCALE
```

### 10.1 Why divide by 2048?

The SDR samples are in device-scale units. Dividing by `ADC_FULL_SCALE` converts them into approximate full-scale units.

Example:

```text
raw sample amplitude = 1024
normalized amplitude = 1024 / 2048 = 0.5
```

### 10.2 Information change

Before:

```text
complex sample in hardware units
```

After:

```text
complex sample as fraction of full scale
```

This prepares the signal for dBFS conversion.

---

## 11. Windowing with Blackman Window

The code uses:

```python
WINDOW = np.blackman(FFT_SIZE).astype(np.float32)
WINDOW_SUM = float(np.sum(WINDOW))
```

Then:

```python
windowed = iq_norm * WINDOW
```

### 11.1 Why windowing is needed

The FFT assumes the input frame repeats forever.

If the first and last sample of the 512-sample frame do not line up smoothly, the FFT sees a sudden jump at the boundary. That artificial jump spreads energy across many frequency bins. This is called spectral leakage.

A window reduces the edge discontinuity by tapering the frame near the start and end.

### 11.2 What the Blackman window does

A Blackman window multiplies the middle samples by values near 1 and the edge samples by values near 0.

Conceptually:

```text
original samples:  [x0, x1, x2, ..., x510, x511]
window values:    [0,  small, ..., large, ..., small, 0]
result:           [x0*0, x1*small, ..., x511*0]
```

### 11.3 Tradeoff

Windowing reduces leakage, but it also changes amplitude. That is why the code later divides by `WINDOW_SUM`.

### 11.4 Information change

Before:

```text
normalized time-domain IQ frame
```

After:

```text
normalized and tapered time-domain IQ frame
```

---

## 12. FFT: Time Domain to Frequency Domain

The code does:

```python
omni_fft = np.fft.fftshift(np.fft.fft(omni_norm * WINDOW))
dir_fft = np.fft.fftshift(np.fft.fft(dir_norm * WINDOW))
```

### 12.1 `np.fft.fft`

`fft` transforms the 512 time samples into 512 complex frequency bins.

Input:

```text
512 complex time samples
```

Output:

```text
512 complex frequency bins
```

Each output bin has:

- magnitude: how strong that frequency component is,
- phase: where that frequency component is in its cycle.

### 12.2 Why the output is complex

A frequency component has both amplitude and phase. A complex number can represent both.

For display and alerts, this project mainly uses magnitude.

### 12.3 `fftshift`

Raw FFT output usually places zero frequency at index 0.

`fftshift` rearranges the bins so the center frequency is in the middle of the array.

After shift:

```text
negative offsets | center | positive offsets
```

For a tuned LO, bins represent frequencies around that LO:

```text
LO - 10 MHz ... LO ... LO + 10 MHz
```

### 12.4 Information change

Before:

```text
512 time samples
```

After:

```text
512 frequency components
```

This is the central transformation that makes a spectrum display possible.

---

## 13. Magnitude and dBFS Conversion

The FFT output is complex. The display needs one strength value per bin.

The code does:

```python
db = 20.0 * np.log10(np.abs(fft_result) / WINDOW_SUM + 1e-20)
```

### 13.1 `np.abs`

For a complex number:

```text
a + jb
```

the magnitude is:

```text
sqrt(a^2 + b^2)
```

This gives the strength of that frequency bin.

### 13.2 Divide by `WINDOW_SUM`

Windowing changes the total amplitude. Dividing by `WINDOW_SUM` compensates for the gain of the window.

This makes the displayed amplitude more consistent.

### 13.3 Add `1e-20`

The log of zero is undefined.

So the code adds a very small number:

```text
0.00000000000000000001
```

This prevents math errors if a bin magnitude is exactly zero.

### 13.4 Why `20 * log10`

The project converts amplitude ratio to dB:

```text
dB = 20 * log10(amplitude_ratio)
```

If working directly with power, the formula would use `10 * log10(power_ratio)`. Here the code uses magnitude/amplitude, so it uses 20.

### 13.5 Clip values

The code clips:

```python
np.clip(db, DB_MIN, DB_MAX)
```

With:

```text
DB_MIN = -140
DB_MAX = 0
```

This prevents extreme values from breaking the display scale.

### 13.6 Cast to float32

The code casts:

```python
astype(np.float32)
```

This reduces memory use and is enough precision for display and alert logic.

### 13.7 Information change

Before:

```text
complex frequency bins
```

After:

```text
real-valued dBFS strength per frequency bin
```

Example:

```text
bin 200 = -82.5 dBFS
bin 201 = -80.1 dBFS
bin 202 = -37.4 dBFS
```

---

## 14. Stitched Wideband Spectrum

One FFT covers only one 20 MHz slice. The full sweep covers many slices.

### 14.1 Per-step write

For step `col`:

```python
s = col * BINS_PER_STEP
e = s + BINS_PER_STEP
self.spectrum_omni[s:e] = omni_db
self.spectrum_dir[s:e] = dir_db
```

Because:

```text
BINS_PER_STEP = FFT_SIZE = 512
```

Each step writes 512 bins into the larger spectrum array.

### 14.2 Frequency axis construction

For each step, the code maps bins from:

```text
center_hz - 10 MHz
to
center_hz + 10 MHz
```

The final frequency axis is stored in GHz for plotting:

```python
self.freq_axis_ghz
```

### 14.3 Information change

Before:

```text
many separate 20 MHz spectra
```

After:

```text
one wide spectrum covering the configured sweep range
```

This wide spectrum is what the spectrum plot uses.

---

## 15. Reference Noise Floor Tracking

Each step returns:

```python
float(np.median(omni_db))
```

### 15.1 Why median?

The median is the middle value.

It is useful for estimating noise floor because a few strong peaks do not affect it as much as they would affect the mean.

Example:

```text
values: -90, -89, -88, -87, -30
mean:   much higher because of -30
median: -88
```

The median better represents the background.

### 15.2 EWMA smoothing

The code updates:

```python
ref_nf = 0.1 * nf + 0.9 * ref_nf
```

EWMA means exponentially weighted moving average.

This gives:

- 10% weight to the new estimate,
- 90% weight to the previous estimate.

Effect:

- noise floor changes smoothly,
- waterfall brightness does not jump violently.

### 15.3 Information change

Before:

```text
full dBFS spectrum
```

After:

```text
one smoothed scalar background estimate
```

This scalar is mainly used for waterfall display normalization.

---

## 16. Spectrum Plot Display

The spectrum plot shows current signal strength versus frequency.

### 16.1 Data used

The code sets:

```python
self.spec_omni_curve.setData(self.freq_axis_ghz, self.spectrum_omni)
self.spec_dir_curve.setData(self.freq_axis_ghz, self.spectrum_dir)
```

X axis:

```text
frequency in GHz
```

Y axis:

```text
signal strength in dBFS
```

### 16.2 What a peak means

A peak means one frequency bin or group of bins has stronger energy than surrounding bins.

It does not automatically identify the source. It only says:

```text
there is more RF energy here
```

### 16.3 Crosshair readout

When the mouse moves over the plot, the code finds the nearest frequency bin and displays:

- frequency,
- dBFS value.

This is just a UI readout. It does not change detection logic.

---

## 17. Waterfall Display

The waterfall shows how the spectrum changes over time.

### 17.1 Create one waterfall line

At the end of a sweep, the full spectrum may have more or fewer bins than the waterfall image width.

The code interpolates to:

```text
WATERFALL_COLS = 800
```

Interpolation means estimating values at new x positions using the existing spectrum data.

### 17.2 Store history

Each new line is appended:

```python
self.waterfall_omni.append(wf_omni_line)
self.waterfall_dir.append(wf_dir_line)
```

The deques keep only the latest `wf_max_lines` lines.

### 17.3 Normalize brightness

The code does:

```python
wf_n = np.clip((wf - nf) / DYNAMIC_RANGE, 0, 1)
```

Where:

- `wf` is dBFS data,
- `nf` is reference noise floor,
- `DYNAMIC_RANGE = 60`.

Example:

```text
noise floor = -90 dBFS
signal      = -60 dBFS

(signal - noise_floor) / 60
= (-60 - -90) / 60
= 30 / 60
= 0.5
```

So this bin becomes middle brightness.

### 17.4 Colormap

The normalized values from 0 to 1 are converted to colors.

Low values are dark/blue. High values become green/yellow/red/white depending on the colormap.

### 17.5 Important display warning

The waterfall brightness is not raw power.

It is:

```text
(dBFS value - estimated noise floor) / display dynamic range
```

So it is a normalized visual intensity.

---

## 18. Monitor Frequencies

The detector does not only draw the whole spectrum. It also watches selected frequencies.

Defaults:

Simple detector:

```text
5.800 GHz
5.900 GHz
5.920 GHz
```

Enhanced detector currently lists:

```text
5.650 GHz
5.900 GHz
5.920 GHz
```

The user can edit monitor frequencies in the UI.

### 18.1 Frequency to bin conversion

To inspect one monitored frequency, the code maps the frequency to the nearest spectrum bin.

Conceptually:

```text
target frequency
-> nearest index in freq_axis_ghz
```

### 18.2 Local search window

The detector does not inspect only one exact bin. It checks nearby bins.

Simple detector uses:

```text
+/- 25 bins
```

Enhanced detector uses:

```text
CFAR_MONITOR_SEARCH_HALF = 25
```

Why:

- signal may not land exactly on the center bin,
- LO/clock error can shift it slightly,
- a real signal can spread over multiple bins,
- a wider search is more forgiving.

### 18.3 Peak extraction

The code takes:

```python
peak = max(local_bins)
```

This collapses many nearby bins into one number:

```text
"strongest signal near this monitored frequency"
```

Information change:

```text
local spectrum window
-> one peak dBFS scalar
```

---

## 19. Simple Detector Alert Logic

This section describes `drone_detector.py`.

### 19.1 Fixed threshold

The user threshold is a dBFS value.

Default:

```text
-40 dBFS
```

The logic is:

```python
above = omni_val > threshold or dir_val > threshold
```

Meaning:

- if RX1 is strong enough, detect,
- or if RX2 is strong enough, detect.

### 19.2 Why use OR?

The two channels may see different strengths.

For example:

- omni may catch broad signal,
- directional may be stronger only when pointed toward source.

Using OR means either channel can trigger.

### 19.3 Persistence counter

Instead of instantly turning alerts on/off every sweep, the code uses a counter.

If detected:

```python
counter = min(counter + 2, WARNING_PERSIST_COUNT + 8)
```

If not detected:

```python
counter = max(counter - 1, 0)
```

Alert is active if:

```python
counter >= WARNING_PERSIST_COUNT
```

With:

```text
WARNING_PERSIST_COUNT = 2
```

### 19.4 Why counter increases by 2 but decreases by 1

This makes alerts appear quickly but disappear more slowly.

Purpose:

- reduce flicker,
- tolerate one missed sweep,
- make intermittent signals easier to notice.

This is not RF amplification. It is time-domain decision smoothing.

### 19.5 UI result

If any monitored frequency is active, the UI displays:

```text
SIGNAL @ 5.xxx GHz
```

If no frequencies are active, the warning label is cleared.

---

## 20. Enhanced Detector: Additional Processing

The enhanced detector adds optional features:

- Welch PSD,
- adaptive baseline,
- CFAR,
- Numba acceleration,
- distance model,
- proximity alert panel,
- optional fastlock.

These features change detection and presentation, but the main spectrum/waterfall display path remains the same single-FFT path.

---

## 21. Welch PSD

PSD means power spectral density.

Welch PSD is a method for estimating spectrum more smoothly by averaging multiple FFTs.

### 21.1 Why Welch exists

A single FFT can be noisy. Random noise causes bin values to jump around.

Welch reduces this variance by:

1. splitting the IQ buffer into overlapping segments,
2. windowing each segment,
3. FFTing each segment,
4. converting each FFT to power,
5. averaging the powers.

### 21.2 Overlap

The enhanced detector uses:

```text
WELCH_OVERLAP_FRAC = 0.5
```

That means adjacent segments overlap by 50%.

If `FFT_SIZE = 512`, hop size is:

```text
512 * (1 - 0.5) = 256 samples
```

So segment starts are:

```text
0, 256, 512, 768, ...
```

### 21.3 Power averaging

For each segment:

```python
X = fftshift(fft(segment * WINDOW))
power = (abs(X) / WINDOW_SUM) ** 2
```

Then:

```python
avg_power = mean(powers)
```

Then dB:

```python
10 * log10(avg_power + 1e-20)
```

Why `10 * log10` here?

Because Welch has already squared the magnitude into power.

### 21.4 Where Welch is used

In the enhanced detector:

- Welch can be used for alert/distance decisions.
- The displayed spectrum/waterfall still uses the single FFT path.

This means the visual plot and alert calculation may not be exactly identical if Welch is enabled.

### 21.5 Information change

Before:

```text
raw IQ buffer
```

After:

```text
averaged dBFS-like spectral estimate
```

Benefit:

- smoother detection values.

Cost:

- more computation,
- less instantaneous response to very short bursts.

---

## 22. Adaptive Baseline

A baseline is a slowly updated estimate of the normal background spectrum.

### 22.1 Why baseline is useful

Not all frequencies have the same background level.

Some areas may always be noisier due to:

- receiver behavior,
- local interference,
- environmental signals,
- hardware response.

A baseline stores "what normal looks like" for each bin.

### 22.2 Baseline update logic

For each bin:

```python
quiet = spec < (baseline + BASELINE_MARGIN_DB)
```

With:

```text
BASELINE_MARGIN_DB = 10.0
BASELINE_ALPHA = 0.02
```

If the bin is quiet:

```python
baseline += 0.02 * (spec - baseline)
```

If the bin is loud:

```python
baseline += 0.002 * (spec - baseline)
```

because loud bins use `BASELINE_ALPHA * 0.1`.

### 22.3 Why loud bins update slower

If a strong signal appears, we do not want the baseline to immediately learn it as "normal".

Updating loud bins slowly helps keep transient signals separate from background.

### 22.4 Important note

In the current enhanced code, the adaptive baseline is maintained as state. CFAR and display logic primarily use spectrum arrays and CFAR thresholds. The baseline is part of advanced tracking state and can support improved noise awareness.

---

## 23. CFAR: Adaptive Threshold Detection

CFAR means Constant False Alarm Rate.

The idea is:

```text
look near a candidate frequency
estimate local background noise
set threshold above that background
detect if candidate peak is above threshold
```

### 23.1 Why fixed threshold is not enough

A fixed threshold like -40 dBFS can be too simple.

Problem examples:

- In a quiet band, -55 dBFS might be suspicious.
- In a noisy band, -45 dBFS might be normal.

CFAR adapts to local conditions.

### 23.2 Cells

CFAR uses cells, which are FFT bins.

In this code:

```text
CFAR_GUARD_CELLS = 4
CFAR_TRAINING_CELLS = 16
CFAR_SCALE_DB = 6.0
```

### 23.3 Guard cells

Guard cells are bins close to the candidate peak that are ignored.

Why ignore them?

Because a real signal can spread into neighboring bins. If those bins were used for noise estimation, the signal would raise its own threshold.

### 23.4 Training cells

Training cells are nearby bins farther away from the candidate. They are used to estimate local noise.

The code uses training cells on both sides.

Conceptually:

```text
training | guard | candidate | guard | training
```

### 23.5 Threshold calculation

The local threshold is:

```text
threshold = noise_estimate + scale
```

With default scale:

```text
scale = 6 dB
```

So if local noise is -70 dBFS:

```text
threshold = -70 + 6 = -64 dBFS
```

A peak at -60 dBFS would pass. A peak at -66 dBFS would not.

### 23.6 Full threshold curve

The enhanced detector can compute a CFAR threshold for every bin. This threshold curve can be drawn on the spectrum plot.

This lets the user see:

- actual spectrum curve,
- adaptive detection line.

### 23.7 Per-monitor CFAR

For each monitored frequency:

1. Convert frequency to nearest bin.
2. Extract +/-25 bins around it.
3. Find local peak.
4. Estimate local noise excluding guard area.
5. Compute threshold.
6. Compare peak to threshold.

Output:

```text
detected: true/false
peak_db: strongest local signal
noise_est_db: local noise estimate
threshold_db: local threshold
```

### 23.8 Information change

Before:

```text
local spectrum window
```

After:

```text
peak, noise estimate, threshold, detected/not detected
```

---

## 24. Hybrid Detection in Enhanced Version

The enhanced detector does not rely only on CFAR.

It uses:

```python
detected = cfar_detected or above_fixed
```

Where:

```python
above_fixed = peak_o > fixed_threshold or peak_d > fixed_threshold
cfar_detected = cfar_det_o or cfar_det_d
```

### 24.1 Meaning

A signal can trigger if:

- CFAR says it is locally unusual,
- or the fixed threshold says it is absolutely strong.

### 24.2 Why use both

CFAR is adaptive, but it can miss signals in some cases.

Fixed threshold is simple, but it can be too insensitive or too sensitive depending on noise.

The hybrid rule favors not missing strong events.

### 24.3 Information change

Before:

```text
separate CFAR and fixed-threshold decisions
```

After:

```text
one final detected/not-detected boolean
```

---

## 25. Alert Persistence Counter in Enhanced Version

Enhanced version uses the same counter style as simple version.

If detected:

```python
counter = min(counter + 2, WARNING_PERSIST_COUNT + 8)
```

If not detected:

```python
counter = max(counter - 1, 0)
```

Alert condition:

```python
counter >= WARNING_PERSIST_COUNT
```

This converts noisy per-sweep booleans into a more stable UI state.

---

## 26. Distance Estimation

The enhanced detector can estimate distance from signal strength.

Important: this is approximate. It depends heavily on calibration, antenna gain, transmitter power, environment, and multipath.

### 26.1 Input to distance model

The code uses the strongest peak:

```python
best_power = max(peak_o, peak_d)
```

So if RX1 sees -50 dBFS and RX2 sees -43 dBFS:

```text
best_power = -43 dBFS
```

### 26.2 Log-distance path loss model

The formula is:

```text
distance = d0 * 10 ^ ((A0 - P) / (10 * n))
```

Where:

- `P` is measured signal level in dBFS.
- `A0` is reference power at reference distance.
- `d0` is reference distance.
- `n` is path loss exponent.

Defaults:

```text
A0 = -30 dBFS
d0 = 10 m
n = 2.2
```

### 26.3 What the formula means

If measured power is close to the reference power, distance is close to reference distance.

If measured power is much weaker, estimated distance increases.

If measured power is stronger, estimated distance decreases.

### 26.4 Clamp

The result is clamped:

```text
minimum = 1 m
maximum = 10000 m
```

This prevents impossible or unusable values from spreading through the UI.

### 26.5 Information change

Before:

```text
signal strength in dBFS
```

After:

```text
estimated distance in meters
```

---

## 27. Kalman Filter Distance Smoothing

A Kalman filter is a mathematical method for combining:

- previous estimate,
- new measurement,
- uncertainty.

The detector uses a simple 1D Kalman filter for distance.

### 27.1 Why distance needs smoothing

Signal strength can jump because of:

- noise,
- multipath,
- antenna angle,
- fading,
- sweep timing,
- hardware gain behavior.

If distance were computed directly from each signal strength measurement, the UI could jump wildly.

### 27.2 State variables

The filter stores:

```text
x = current distance estimate
P = estimate uncertainty
```

It also uses:

```text
Q = process noise = 25
R = measurement noise = 400
```

### 27.3 Prediction step

```python
P_pred = P + Q
```

This says uncertainty grows slightly over time.

### 27.4 Kalman gain

```python
K = P_pred / (P_pred + R)
```

Kalman gain controls how much to trust the new measurement.

- Larger `K`: trust new measurement more.
- Smaller `K`: trust old estimate more.

### 27.5 Correction step

```python
x = x + K * (z - x)
```

Where:

- `z` is the new raw distance measurement,
- `x` is the previous filtered estimate.

This moves `x` toward `z`, but usually not all the way.

### 27.6 Uncertainty update

```python
P = (1 - K) * P_pred
```

After using a measurement, uncertainty usually decreases.

### 27.7 Confidence calculation

```python
confidence = 1 - (P / (P + 100))
```

This maps uncertainty into a value between 0 and 1.

Higher confidence means lower uncertainty.

### 27.8 Information change

Before:

```text
raw distance estimate, jumpy
```

After:

```text
smoothed distance estimate and confidence
```

---

## 28. Proximity Hysteresis

Hysteresis means using different thresholds for entering and leaving a state.

The enhanced detector uses:

```text
enter boundary: distance < 95 m
exit boundary:  distance > 120 m
```

### 28.1 Why not use one threshold?

If the distance estimate hovers near 100 m, a single threshold could cause:

```text
alert on, alert off, alert on, alert off
```

Hysteresis prevents this.

### 28.2 State behavior

If currently outside:

```text
enter only when distance < 95 m
```

If currently inside:

```text
exit only when distance > 120 m
```

### 28.3 Information change

Before:

```text
continuous distance estimate
```

After:

```text
stable inside/outside boundary state
```

---

## 29. Proximity Alert Panel

The enhanced detector pushes each monitor result into `ProximityAlertPanel`.

Input:

```text
frequency
signal dBFS
distance, optional
confidence
```

### 29.1 AlertEngine history

`AlertEngine` keeps recent signal values per frequency.

This lets it determine trend.

### 29.2 Trend detection

The engine looks at recent signal history.

It compares the average of the first half of the recent window to the average of the second half.

If recent values are stronger by more than 1.5 dB:

```text
trend = RISING
```

If weaker by more than 1.5 dB:

```text
trend = FALLING
```

Otherwise:

```text
trend = STABLE
```

If there is not enough history:

```text
trend = UNKNOWN
```

### 29.3 Threat levels

Threat levels are:

```text
NONE
DETECTED
APPROACHING
CRITICAL
```

The engine uses:

- signal threshold,
- distance,
- boundary hysteresis,
- trend.

### 29.4 Threat rule

If signal is below engine detection threshold:

```text
NONE
```

If inside boundary or closer than critical distance:

```text
CRITICAL
```

If closer than approaching distance:

```text
APPROACHING
```

Otherwise:

```text
DETECTED
```

If threat is `APPROACHING` and trend is `RISING`, it can escalate to `CRITICAL`.

### 29.5 UI output

The panel displays:

- frequency,
- dBFS,
- distance,
- trend arrow,
- threat label,
- color,
- banner for worst active alert,
- flashing border for critical rows.

These are UI states derived from the processed signal information.

---

## 30. Optional Fastlock

The enhanced detector includes optional AD9361 fastlock support.

Fastlock is a hardware feature that stores precomputed tuning profiles so frequency changes can happen faster.

In the current feature flags:

```text
use_fastlock = False
```

So it is off by default.

If enabled and supported, it affects retuning speed, not the mathematical meaning of FFT bins.

---

## 31. Optional Numba Acceleration

Numba is a Python package that can compile numerical Python code to faster machine code.

The enhanced detector can use it for CFAR threshold calculation.

Important:

- Numba should not change the detection meaning.
- It only changes how fast the CFAR calculation runs.

If Numba is not installed, NumPy implementation is used.

---

## 32. Full Information Transformation Table

| Stage | Input | Operation | Output |
|---|---|---|---|
| Radio propagation | RF source energy | travels through space | RF field at antenna |
| Antenna | RF field | electromagnetic-to-electrical conversion | tiny voltage/current |
| SDR tuning | broadband RF | select LO frequency | 20 MHz slice |
| Mixing | RF slice | shift to baseband | low-frequency IQ analog signal |
| ADC | analog baseband | sample at 20 MS/s | digital samples |
| Driver | ADC stream | package channel data | `raw` from `sdr.rx()` |
| Channel unpack | backend-specific raw object | convert/extract arrays | `omni_iq`, `dir_iq` |
| Frame length | variable IQ length | pad/truncate | 512-sample frame |
| Normalize | ADC-scale IQ | divide by 2048 | full-scale fraction IQ |
| Window | normalized IQ | multiply by Blackman window | tapered IQ |
| FFT | time-domain IQ | Fourier transform | complex frequency bins |
| Shift | raw FFT order | `fftshift` | centered frequency bins |
| Magnitude | complex bins | absolute value | amplitude per bin |
| dBFS | amplitude | `20*log10(...)` | dBFS per bin |
| Clip | dBFS values | clamp -140 to 0 | bounded dBFS spectrum |
| Stitch | per-step spectrum | write into full vector | wideband spectrum |
| Plot | spectrum vector | pyqtgraph curve update | spectrum display |
| Waterfall | spectrum history | interpolate + normalize | color image |
| Monitor extract | wideband spectrum | nearest bin + local max | peak near frequency |
| Fixed threshold | peak | compare to slider | detected boolean |
| CFAR | local bins | adaptive threshold | detected boolean |
| Hybrid decision | booleans | OR logic | final detection |
| Persistence | final detection | counter update | stable alert state |
| Distance | peak dBFS | path loss model | raw distance |
| Kalman | raw distance | smoothing | filtered distance/confidence |
| Hysteresis | filtered distance | enter/exit thresholds | inside/outside state |
| Alert engine | signal/distance/history | classify threat | alert object |
| UI widget | alert object | labels/colors/flashing | operator-visible alert |

---

## 33. What Counts as "Boosting" in This Project

The word "boosting" can mean different things. The project has several mechanisms that make signals more visible or alerts more likely, but they are not all physical amplification.

### 33.1 Physical or hardware boost

Hardware gain:

```python
rx_hardwaregain_chan0
rx_hardwaregain_chan1
```

This actually amplifies the received signal in the SDR chain.

### 33.2 Visual boost

Waterfall normalization:

```python
(wf - noise_floor) / dynamic_range
```

This does not change the signal. It changes how bright it appears.

### 33.3 Detection boost

Hybrid logic:

```python
CFAR detection OR fixed-threshold detection
```

This makes detection more permissive than either method alone.

### 33.4 Temporal boost

Counter increment:

```python
counter += 2
```

This makes a positive detection quickly become an active alert.

### 33.5 Threat boost

Trend escalation:

```text
APPROACHING + RISING -> CRITICAL
```

This changes UI severity based on trend.

---

## 34. What Counts as Suppression or Stabilization

The project also has mechanisms that reduce noise, false triggers, and flicker.

### 34.1 Blackman window

Reduces spectral leakage.

### 34.2 Median noise estimate

Reduces influence of isolated peaks on noise-floor estimate.

### 34.3 EWMA noise smoothing

Prevents waterfall brightness from jumping.

### 34.4 Welch PSD

Averages multiple FFT powers to reduce random variation.

### 34.5 CFAR guard cells

Prevents the candidate signal from contaminating its own noise estimate.

### 34.6 CFAR training cells

Estimate local noise instead of using global assumptions.

### 34.7 Persistence counter decay

Avoids instant alert flicker.

### 34.8 Kalman filter

Smooths jumpy distance estimates.

### 34.9 Hysteresis

Prevents boundary on/off chatter near distance thresholds.

---

## 35. Display Path vs Alert Path

This distinction is important.

### 35.1 Display path

The spectrum and waterfall are based on:

```text
single FFT per sweep step
stitched into spectrum_omni and spectrum_dir
```

### 35.2 Alert path in simple version

The simple alert path uses:

```text
local max near monitor frequency
fixed threshold
persistence counter
```

### 35.3 Alert path in enhanced version

The enhanced alert path can use:

```text
Welch PSD
CFAR
fixed threshold fallback
persistence counter
distance model
Kalman filter
hysteresis
alert engine trend/threat logic
```

So the display and alert may be related but not always identical.

---

## 36. Step-by-Step Example

Assume the detector is monitoring 5.900 GHz.

### 36.1 RF arrives

A signal exists near 5.900 GHz. The antenna receives it as a tiny voltage.

### 36.2 SDR tunes

During one sweep step, the SDR tunes to a center frequency whose 20 MHz slice includes 5.900 GHz.

### 36.3 IQ captured

The SDR returns complex IQ samples.

### 36.4 FFT calculated

The code converts 512 samples into 512 frequency bins.

### 36.5 dBFS spectrum created

The bin near 5.900 GHz might show:

```text
-36 dBFS
```

Nearby noise might be:

```text
-75 dBFS
```

### 36.6 Spectrum display updates

The spectrum curve shows a peak around 5.900 GHz.

### 36.7 Waterfall updates

A bright pixel/stripe appears at that frequency over time.

### 36.8 Monitor peak extraction

The detector checks +/-25 bins around 5.900 GHz and finds the strongest bin:

```text
peak = -36 dBFS
```

### 36.9 Threshold decision

If fixed threshold is -40 dBFS:

```text
-36 > -40
```

So fixed-threshold detection is true.

### 36.10 Counter update

If previous counter was 0:

```text
new counter = 2
```

Since alert threshold is 2, alert becomes active.

### 36.11 Enhanced distance estimate

If enhanced distance is on, the code estimates distance from -36 dBFS using the path loss model.

### 36.12 UI alert

The UI displays signal/proximity information for 5.900 GHz.

---

## 37. Common Misunderstandings

### 37.1 "A spectrum peak means it is definitely a drone."

Not necessarily.

A peak means RF energy exists at that frequency. Classification depends on context, known frequencies, signal behavior, and operator interpretation.

### 37.2 "Waterfall color is absolute power."

Not exactly.

Waterfall color is normalized relative to estimated noise floor and display dynamic range.

### 37.3 "dBFS is dBm."

No.

dBFS is relative to ADC full scale. dBm is absolute power relative to 1 milliwatt. This project uses dBFS.

### 37.4 "RX2 always means directional antenna."

Only if true dual-channel hardware is working.

Otherwise RX2 is a mirrored copy of RX1.

### 37.5 "CFAR always beats fixed threshold."

Not always.

CFAR adapts to local noise, but fixed threshold can catch strong signals that CFAR might not flag in some conditions. The enhanced detector uses both.

---

## 38. Code Location Map

### 38.1 Acquisition and FFT

Files:

- `drone_detector.py`
- `drone_detector_enhanced.py`

Main method:

- `SweepWorker._do_one_step`

### 38.2 Sweep completion and alert update

Files:

- `drone_detector.py`
- `drone_detector_enhanced.py`

Main method:

- `_on_sweep_done`

### 38.3 Display update

Files:

- `drone_detector.py`
- `drone_detector_enhanced.py`

Main method:

- `_update_display`

### 38.4 Welch PSD

File:

- `drone_detector_enhanced.py`

Main function:

- `_welch_psd_db`

### 38.5 CFAR

File:

- `drone_detector_enhanced.py`

Main functions:

- `cfar_threshold`
- `cfar_detect_at_bin`

### 38.6 Distance model

File:

- `drone_detector_enhanced.py`

Main class:

- `DistanceEstimator`

### 38.7 Proximity alert classification

Files:

- `proximity_alert/engine.py`
- `proximity_alert/widget.py`

Main classes:

- `AlertEngine`
- `ProximityAlertPanel`

---

## 39. Complete One-Sweep Summary

One complete sweep does this:

1. Start at the configured lowest frequency.
2. Tune the SDR local oscillator.
3. Wait for hardware to settle.
4. Read IQ samples.
5. Extract or mirror RX channels.
6. Pad/truncate to 512 samples.
7. Divide by ADC full scale.
8. Apply Blackman window.
9. Run FFT.
10. Shift FFT bins.
11. Convert complex bins to magnitude.
12. Convert magnitude to dBFS.
13. Clip dBFS values.
14. Store the 512-bin slice in the wide spectrum.
15. Update noise-floor estimate.
16. Move to the next 20 MHz frequency step.
17. Repeat until the whole configured band is covered.
18. Interpolate spectrum to waterfall width.
19. Append waterfall line.
20. For each monitor frequency, find local peak.
21. Run fixed threshold and/or CFAR decision.
22. Update persistence counter.
23. Optionally estimate distance.
24. Optionally smooth distance with Kalman filter.
25. Optionally update proximity boundary state.
26. Update spectrum plot.
27. Update waterfall plot.
28. Update monitor mini-plots.
29. Update warning label and proximity alert panel.

---

## 40. Final Mental Model

The detector is a chain of transformations.

The physical world gives:

```text
radio energy
```

The SDR converts it into:

```text
complex IQ samples
```

The FFT converts those into:

```text
frequency bins
```

dBFS conversion turns those into:

```text
human-readable signal strengths
```

The sweep stitcher turns small slices into:

```text
wideband spectrum
```

The display turns that into:

```text
spectrum and waterfall graphics
```

The detection logic turns monitored peaks into:

```text
alert or no alert
```

The enhanced logic can further turn signal strength into:

```text
estimated distance, confidence, trend, and threat level
```

That is the complete information path from incoming signal to what the operator sees.
