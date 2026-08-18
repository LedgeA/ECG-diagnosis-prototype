#!/usr/bin/env bash
# run_ph_synthesis_realistic.sh
# Batch generates synthetic 3x4 ECG images with advanced noise, creases,
# demographic headers, and color adjustments to closely resemble real-world paper records.

set -e

# Dynamically resolve workspace path to current script location
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DSP_DIR="$WORKSPACE/records100_dsp"
OUTPUT_IMG_DIR="$WORKSPACE/synthetic_ph_3x4_dataset_realistic"

# Locate generator script in ecg-image-kit
GENERATOR_SCRIPT=$(find "$WORKSPACE/ecg-image-kit" -name "gen_ecg_images_from_data_batch.py" | head -n 1)

if [ -z "$GENERATOR_SCRIPT" ]; then
    echo "Error: Could not find gen_ecg_images_from_data_batch.py in $WORKSPACE/ecg-image-kit"
    exit 1
fi

if [ ! -d "$INPUT_DSP_DIR" ]; then
    echo "Error: Input DSP directory $INPUT_DSP_DIR does not exist."
    echo "Please run preprocess_ph_dsp.py first."
    exit 1
fi

echo "Using generator script at: $GENERATOR_SCRIPT"
mkdir -p "$OUTPUT_IMG_DIR"

for subfolder in $(ls "$INPUT_DSP_DIR" | sort); do
    IN_SUB="$INPUT_DSP_DIR/$subfolder"
    OUT_SUB="$OUTPUT_IMG_DIR/$subfolder"
    
    if [ -d "$IN_SUB" ]; then
        echo "Processing subfolder $subfolder with realistic settings..."
        mkdir -p "$OUT_SUB"
        
        # Configuration mapping to realistic ECG attributes:
        # --print_header: Outputs patient/lead details to mimic clinical printouts.
        # --standard_grid_color 2: Uses pink grid lines (instead of bright red) to resemble typical thermal paper.
        # --calibration_pulse 1.0: Ensures calibration pulses are visible on the paper.
        # --wrinkles: Applies physical paper folds and shadow texture.
        # -ca, -nv, -nh: Defines angle (45 deg) and crease counts.
        # --augment: Simulates photo scan noise, crop, color temperature, and angle skews.
        # -rot 3: Tilts the page slightly (3 degrees) to mimic a handheld or slightly misaligned photo/scan.
        # -noise 30: Adds sensor grain/noise.
        # -t 5000: Sets color temperature to 5000K (warmer daylight/fluorescent light).
        python3 "$GENERATOR_SCRIPT" \
            -i "$IN_SUB" \
            -o "$OUT_SUB" \
            -se 10 \
            --store_config 2 \
            --lead_bbox \
            --lead_name_bbox \
            --mask_unplotted_samples \
            --print_header \
            --standard_grid_color 2 \
            --calibration_pulse 1.0 \
            --pad_inches 1 \
            --wrinkles \
            -ca 45 \
            -nv 8 \
            -nh 8 \
            --augment \
            -rot 1 \
            -noise 20 \
            -c 0.005 \
            -t 5000 \
            --deterministic_rot \
            --deterministic_noise \
            --deterministic_crop \
            --deterministic_temp
            
        echo "Finished subfolder $subfolder"
    fi
done

echo "Realistic synthetic 3x4 ECG dataset successfully generated at $OUTPUT_IMG_DIR"
