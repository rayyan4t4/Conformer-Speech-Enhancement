import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def load_audio_mono(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Loads audio file, converts to mono, and resamples to target_sr."""
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path)
    except Exception:
        import scipy.io.wavfile as wavfile
        sr, audio = wavfile.read(file_path)

    # Convert integer types to float [-1.0, 1.0]
    if audio.dtype in [np.int16, np.int32]:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    else:
        audio = audio.astype(np.float32)

    # Average channels if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=-1)

    # Resample if needed
    if sr != target_sr:
        num_samples = int(len(audio) * float(target_sr) / sr)
        audio = scipy.signal.resample(audio, num_samples)
        sr = target_sr

    return audio, sr


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000):
    """Saves normalized audio to WAV file."""
    try:
        import soundfile as sf
        sf.write(file_path, audio, sr)
    except Exception:
        import scipy.io.wavfile as wavfile
        scaled = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        wavfile.write(file_path, sr, scaled)


def generate_spectrogram_plot(
    noisy_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    clean_audio: np.ndarray = None,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256
) -> plt.Figure:
    """
    Creates a publication-quality side-by-side comparison spectrogram plot.
    """
    cols = 3 if clean_audio is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 3.5), sharey=True)

    tracks = [("Noisy Input Audio", noisy_audio), ("Enhanced Audio (Conformer U-Net)", enhanced_audio)]
    if clean_audio is not None:
        tracks.append(("Clean Reference Speech", clean_audio))

    for i, (title, wave) in enumerate(tracks):
        f, t, Sxx = scipy.signal.spectrogram(
            wave,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            scaling='spectrum'
        )
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
        im = axes[i].pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma', vmin=-80, vmax=0)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Time (s)", fontsize=10)
        if i == 0:
            axes[i].set_ylabel("Frequency (Hz)", fontsize=10)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Magnitude (dB)")
    return fig


def mix_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mixes clean speech and noise at a target Signal-to-Noise Ratio (SNR)."""
    # Match lengths
    if len(noise) < len(clean):
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        noise = noise[:len(clean)]

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    noisy = clean + scale * noise
    # Prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 0.99:
        noisy = noisy * (0.95 / max_val)
    return noisy
