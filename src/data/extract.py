"""Convert raw FMCW radar samples into RTM, DTM, and ATM tensors.

The extractor assumes the raw acquisition folder contains a `radar.npy`
recording and a `config.json` metadata file. It writes one compressed NumPy
array per acquisition folder as `preprocessed.npz`.
"""

import json
import os

import numpy as np

from src.config import (
    DATA_DIR,
    D_BY_LAMBDA,
    EPSILON,
    GESTURES_PER_BATCH,
    MAX_ANGLE_DEG,
    NUM_BEAMS,
)
from src.helpers.DigitalBeamForming import DigitalBeamForming
from src.helpers.DopplerAlgo import DopplerAlgo


def process_batch(dirpath):
    """Process one raw gesture batch directory into the radar feature tensor."""
    print(f"Processing gesture batch in: {dirpath}")

    with open(os.path.join(dirpath, "config.json"), "r") as file:
        json.load(file)["device_config"]["fmcw_single_shape"]

    raw_data = np.load(os.path.join(dirpath, "radar.npy")) / 4095.0
    batch_nframes, batch_nsensors, batch_nchirps, batch_nsamples = raw_data.shape

    doppler_algo = DopplerAlgo(batch_nsamples, batch_nchirps, batch_nsensors)

    d_transformed = np.zeros(
        (batch_nframes, batch_nsensors, batch_nsamples, batch_nchirps * 2),
        dtype=complex,
    )

    for f in range(batch_nframes):
        for s in range(batch_nsensors):
            d_transformed[f, s] = doppler_algo.compute_doppler_map(raw_data[f, s], s)

    AZIMUTH_ANTENNAS = [0, 2]
    dbf_algo = DigitalBeamForming(
        num_antennas=len(AZIMUTH_ANTENNAS),
        num_beams=NUM_BEAMS,
        max_angle_degrees=MAX_ANGLE_DEG,
        d_by_lambda=D_BY_LAMBDA,
    )

    atm_all_frames = np.zeros((batch_nframes, NUM_BEAMS))
    for f in range(batch_nframes):
        rd_spectrum = np.transpose(d_transformed[f, AZIMUTH_ANTENNAS], (1, 2, 0))
        rd_beam_formed = dbf_algo.run(rd_spectrum)
        atm_all_frames[f] = np.mean(np.abs(rd_beam_formed), axis=(0, 1))

    cut = batch_nframes % GESTURES_PER_BATCH
    window = batch_nframes // GESTURES_PER_BATCH

    d_trans_reshaped = d_transformed[cut:].reshape(
        GESTURES_PER_BATCH,
        window,
        batch_nsensors,
        batch_nsamples,
        batch_nchirps * 2,
    )
    atm_reshaped = atm_all_frames[cut:].reshape(GESTURES_PER_BATCH, window, NUM_BEAMS)

    preproc_data = np.zeros((GESTURES_PER_BATCH, window, batch_nsamples, 3))
    preproc_data[..., 0] = np.mean(np.abs(d_trans_reshaped), axis=(2, 4))
    preproc_data[..., 1] = np.mean(np.abs(d_trans_reshaped), axis=(2, 3))
    preproc_data[..., 2] = atm_reshaped

    preproc_data = 10 * np.log10(preproc_data + EPSILON)

    out_path = os.path.join(dirpath, "preprocessed.npz")
    np.savez_compressed(out_path, data=preproc_data)


def extract_all():
    """Loop through the private data tree and run preprocessing on each folder."""
    for dirpath, _, filenames in os.walk(DATA_DIR):
        if "radar.npy" in filenames:
            process_batch(dirpath)


if __name__ == "__main__":
    extract_all()
    print("Feature extraction complete.")