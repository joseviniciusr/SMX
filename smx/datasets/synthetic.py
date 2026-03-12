import numpy as np
import pandas as pd


def gaussian_peak_model(x, center, amplitude, width):
    """
    Generate a one-dimensional Gaussian peak.

    Implements the equation:
    g(x) = A * exp(-(x - c)² / (2σ²))

    Parameters
    ----------
    x : array_like
        Spectral axis (wavelengths, energy, channels).
    center : float
        Central position of the peak (same units as x).
    amplitude : float
        Maximum height of the peak (intensity at the center).
    width : float
        Standard deviation (σ) of the peak — controls spread/width.

    Returns
    -------
    ndarray
        Array with the Gaussian peak evaluated at each point of x.

    Notes
    -----
    - For XRF: use a small width (~5–15) to simulate narrow lines.
    - For Vis-NIR: use a larger width (~20–50) for broad absorption bands.
    """
    return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))


def _generate_single_spectrum(
    x,
    peaks,
    amplitude_mean=1.0,
    amplitude_std=0.1,
    width_mean=15.0,
    width_std=2.0,
    noise_std=0.02,
):
    """
    Generate a single spectrum by summing Gaussian peaks with variability + noise.

    Internal helper for generate_synthetic_spectral_data.

    Parameters
    ----------
    x : ndarray
        Spectral axis.
    peaks : list of float
        Peak centre positions to add.
    amplitude_mean, amplitude_std : float
        Mean and standard deviation of peak amplitude.
    width_mean, width_std : float
        Mean and standard deviation of peak width.
    noise_std : float
        Standard deviation of the Gaussian baseline noise.

    Returns
    -------
    ndarray
        Synthetic spectrum (baseline noise + peaks).
    """
    # Baseline: white Gaussian noise
    spectrum = np.random.normal(0, noise_std, len(x))

    # Add each peak with random variability
    for center in peaks:
        amp = np.random.normal(amplitude_mean, amplitude_std)
        width = np.random.normal(width_mean, width_std)
        spectrum += gaussian_peak_model(x, center, amp, width)

    return spectrum


def generate_synthetic_spectral_data(
    classes_config,
    n_points=500,
    x_min=0,
    x_max=1000,
    seed=None,
):
    """
    Generate a synthetic spectral dataset for multiple classes.

    Returns a DataFrame where:
    - First column: ``'Class'`` (values defined by the user: 'A', 'B', 'C', …).
    - Remaining columns: spectral variables (intensity values).
    - Rows: individual samples.

    Parameters
    ----------
    classes_config : list of dict
        List of dicts, each defining one class. Supported keys:

        - ``'name'`` (str): class label (e.g. ``'A'``, ``'B'``, ``'Soil'``).
        - ``'n_samples'`` (int): number of samples to generate.
        - ``'peaks'`` (list of float): peak centre positions on the spectral axis.
        - ``'amplitude_mean'`` (float, optional, default ``1.0``): mean peak amplitude.
        - ``'amplitude_std'`` (float, optional, default ``0.1``): std dev of amplitude.
        - ``'width_mean'`` (float, optional, default ``15.0``): mean peak width (σ).
        - ``'width_std'`` (float, optional, default ``2.0``): std dev of peak width.
        - ``'noise_std'`` (float, optional, default ``0.02``): std dev of baseline noise.

        Example::

            [
                {
                    'name': 'A',
                    'n_samples': 50,
                    'peaks': [250, 550, 700, 850],
                    'amplitude_mean': 1.0,
                    'width_mean': 15.0,
                },
                {
                    'name': 'B',
                    'n_samples': 50,
                    'peaks': [250, 700, 850],
                    'amplitude_mean': 1.2,
                    'width_mean': 20.0,
                },
            ]

    n_points : int, default ``500``
        Number of points on the spectral axis (resolution).
    x_min, x_max : float, default ``0``, ``1000``
        Limits of the spectral axis (e.g. 400–1000 nm for Vis-NIR,
        0–40 keV for XRF).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        Synthetic spectral dataset.

        - Column 0: ``'Class'`` (str — class name from *classes_config*).
        - Columns 1 … n_points: spectral intensities named after x-axis values.
        - Shape: ``(total_samples, n_points + 1)``.
    """
    if seed is not None:
        np.random.seed(seed)

    x_axis = np.linspace(x_min, x_max, n_points)

    spectra_list = []
    labels_list = []

    for config in classes_config:
        class_name = config["name"]
        n_samples = config["n_samples"]
        peaks = config["peaks"]
        amplitude_mean = config.get("amplitude_mean", 1.0)
        amplitude_std = config.get("amplitude_std", 0.1)
        width_mean = config.get("width_mean", 15.0)
        width_std = config.get("width_std", 2.0)
        noise_std = config.get("noise_std", 0.02)

        for _ in range(n_samples):
            spectrum = _generate_single_spectrum(
                x_axis,
                peaks,
                amplitude_mean,
                amplitude_std,
                width_mean,
                width_std,
                noise_std,
            )
            spectra_list.append(spectrum)
            labels_list.append(class_name)

    spectra_array = np.array(spectra_list)
    column_names = x_axis.astype(str).tolist()
    df = pd.DataFrame(spectra_array, columns=column_names)
    df.insert(0, "Class", labels_list)

    return df
