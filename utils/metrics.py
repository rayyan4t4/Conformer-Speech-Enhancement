import numpy as np

def compute_si_sdr(estimated: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """
    Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    """
    est = estimated - np.mean(estimated)
    ref = reference - np.mean(reference)

    ref_energy = np.sum(ref ** 2) + eps
    dot = np.sum(est * ref)
    s_target = (dot / ref_energy) * ref
    e_noise = est - s_target

    target_energy = np.sum(s_target ** 2) + eps
    noise_energy = np.sum(e_noise ** 2) + eps

    return float(10 * np.log10(target_energy / noise_energy))


def compute_snr(estimated: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Computes standard Signal-to-Noise Ratio (SNR) in dB."""
    noise = estimated - reference
    sig_power = np.sum(reference ** 2) + eps
    noise_power = np.sum(noise ** 2) + eps
    return float(10 * np.log10(sig_power / noise_power))


def compute_pesq_score(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """
    Computes Wideband PESQ score (returns float between -0.5 and 4.5).
    Safely falls back if pesq package is not installed.
    """
    try:
        from pesq import pesq
        mode = 'wb' if sr == 16000 else 'nb'
        return float(pesq(sr, clean, enhanced, mode))
    except ImportError:
        return float('nan')
    except Exception:
        return float('nan')


def compute_stoi_score(clean: np.ndarray, enhanced: np.ndarray, sr: int = 16000) -> float:
    """
    Computes STOI (Short-Time Objective Intelligibility) score (0.0 to 1.0).
    Safely falls back if pystoi package is not installed.
    """
    try:
        from pystoi import stoi
        return float(stoi(clean, enhanced, sr, extended=False))
    except ImportError:
        return float('nan')
    except Exception:
        return float('nan')


def evaluate_batch_metrics(clean_batch: np.ndarray, noisy_batch: np.ndarray, enh_batch: np.ndarray, sr: int = 16000) -> dict:
    """
    Calculates average SI-SDR, SNR improvement, PESQ, and STOI over a batch of audio samples.
    """
    si_sdr_improvements = []
    pesq_scores = []
    stoi_scores = []

    for c, n, e in zip(clean_batch, noisy_batch, enh_batch):
        sdr_noisy = compute_si_sdr(n, c)
        sdr_enh = compute_si_sdr(e, c)
        si_sdr_improvements.append(sdr_enh - sdr_noisy)

        p = compute_pesq_score(c, e, sr)
        if not np.isnan(p):
            pesq_scores.append(p)

        s = compute_stoi_score(c, e, sr)
        if not np.isnan(s):
            stoi_scores.append(s)

    results = {
        "Delta_SI_SDR_dB": float(np.mean(si_sdr_improvements)) if si_sdr_improvements else 0.0,
        "PESQ_WB": float(np.mean(pesq_scores)) if pesq_scores else float('nan'),
        "STOI": float(np.mean(stoi_scores)) if stoi_scores else float('nan')
    }
    return results
