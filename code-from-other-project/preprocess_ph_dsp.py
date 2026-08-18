#!/usr/bin/env python3
"""
preprocess_ph_dsp.py
Applies Philippine-standard digital signal processing (DSP) to PTB-XL records100.
Filters signals for diagnostic bandwidth (0.05 Hz - 35 Hz for Fs=100Hz) to remove
baseline wander and muscle tremor artifacts prior to synthetic image rendering.
"""

import os
import glob
import numpy as np
import scipy.signal as signal
import wfdb

# Dynamically resolve workspace paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "records100")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "records100_dsp")

def apply_ph_filters(sig, fs=100):
    """
    Applies zero-phase digital filtering conforming to Philippine clinical standards.
    For Fs = 100 Hz, Nyquist limit is 50 Hz.
    High-pass cutoff: 0.05 Hz (Baseline wander removal)
    Low-pass cutoff: 35.0 Hz (Muscle tremor filter standard)
    """
    nyquist = 0.5 * fs
    low_cutoff = 0.05 / nyquist
    high_cutoff = 35.0 / nyquist
    
    b, a = signal.butter(2, [low_cutoff, high_cutoff], btype='bandpass')
    filtered_sig = signal.filtfilt(b, a, sig, axis=0)
    return filtered_sig

def process_wfdb_directory(input_base, output_base):
    if not os.path.exists(input_base):
        print(f"Error: Directory '{input_base}' does not exist.")
        print("Ensure 'records100' is located inside the same folder as this script.")
        return

    record_paths = glob.glob(os.path.join(input_base, "**", "*.hea"), recursive=True)
    print(f"Found {len(record_paths)} WFDB records for DSP preprocessing in {input_base}...")

    for hea_path in record_paths:
        record_dir = os.path.dirname(hea_path)
        record_name = os.path.basename(hea_path).replace(".hea", "")
        
        rel_path = os.path.relpath(record_dir, input_base)
        target_dir = os.path.join(output_base, rel_path)
        os.makedirs(target_dir, exist_ok=True)
        
        rel_record_path = os.path.join(record_dir, record_name)
        
        try:
            record = wfdb.rdrecord(rel_record_path)
            filtered_p_signal = apply_ph_filters(record.p_signal, fs=record.fs)
            
            wfdb.wrsamp(
                record_name=record_name,
                fs=record.fs,
                units=record.units,
                sig_name=record.sig_name,
                p_signal=filtered_p_signal,
                fmt=record.fmt,
                adc_gain=record.adc_gain,
                baseline=record.baseline,
                write_dir=target_dir
            )
        except Exception as e:
            print(f"Error processing record {rel_record_path}: {e}")

if __name__ == "__main__":
    process_wfdb_directory(INPUT_DIR, OUTPUT_DIR)
    print("DSP Preprocessing complete. Filtered records saved to records100_dsp.")