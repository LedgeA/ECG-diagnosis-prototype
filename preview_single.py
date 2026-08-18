#!/usr/bin/env python3
"""Single-record preview script to demonstrate rendering & augmentation on a random record.

Usage:
    python preview_single.py
"""
import os
import sys
import random
import subprocess
from pathlib import Path

# Add pipeline directory to sys.path
pipeline_dir = Path(__file__).resolve().parent / "pipeline"
sys.path.insert(0, str(pipeline_dir))

import config as C
import stage5_augment as stage5
import patch_kit

def main():
    print("=== Single Record Augmentation Preview ===")
    
    # 1. Ensure patch is applied
    if not patch_kit.is_patched():
        patch_kit.main()

    # 2. Find all staged records
    staged_dir = C.STAGED
    dat_files = list(staged_dir.rglob("*.dat"))
    if not dat_files:
        sys.exit(f"Error: No staged .dat files found under {staged_dir}. Run pipeline stages 1-3 first.")
    
    # Pick a random staged record
    chosen_dat = random.choice(dat_files)
    stem = chosen_dat.stem
    chunk_dir = chosen_dat.parent
    cls_name = chunk_dir.parent.name
    
    print(f"Selected Record: {stem} (Class: {cls_name}) from {chunk_dir}")
    
    # 3. Create output directory build/preview
    preview_dir = C.BUILD / "preview"
    preview_rendered = preview_dir / "rendered"
    preview_augmented = preview_dir / "augmented"
    
    preview_rendered.mkdir(parents=True, exist_ok=True)
    preview_augmented.mkdir(parents=True, exist_ok=True)
    
    # Render clean image using ecg-image-kit for this specific file
    # We create a temporary input directory containing just this record pair (.dat and .hea)
    single_input_dir = preview_dir / "single_input"
    single_input_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy record files to single_input_dir
    import shutil
    shutil.copy2(chosen_dat, single_input_dir / chosen_dat.name)
    shutil.copy2(chosen_dat.with_suffix(".hea"), single_input_dir / chosen_dat.with_suffix(".hea").name)
    
    print(f"\n[1/2] Rendering clean sheet using ecg-image-kit...")
    cmd = [
        str(C.VENV_PY), "gen_ecg_images_from_data_batch.py",
        "-i", str(single_input_dir),
        "-o", str(preview_rendered),
        "-se", str(random.randint(1000, 9999)),
        *C.KIT_RENDER_FLAGS,
    ]
    proc = subprocess.run(cmd, cwd=str(C.KIT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"Rendering failed: {proc.stderr}")
    
    # Locate clean rendered image
    rendered_pngs = list(preview_rendered.glob(f"{stem}*.png"))
    if not rendered_pngs:
        sys.exit("Error: Could not find clean rendered PNG.")
    clean_png = rendered_pngs[0]
    print(f"  Clean rendered sheet: {clean_png}")

    # 4. Generate 3 different random augmentation variations
    print(f"\n[2/2] Applying Stage 5 paper/scan augmentations (3 variations)...")
    augmented_paths = []
    for var in range(1, 4):
        dst_jpeg = preview_augmented / f"{stem}_augmented_var{var}.jpg"
        seed_parts = (stem, "preview_var", var, random.random())
        stage5.augment(clean_png, dst_jpeg, seed_parts)
        augmented_paths.append(dst_jpeg)
        print(f"  Augmented variant {var}: {dst_jpeg}")
        
    print("\n=== Preview Completed Successfully ===")
    print("\nYou can inspect the output files at:")
    print(f"  Clean PNG : file://{clean_png.resolve()}")
    for i, path in enumerate(augmented_paths, start=1):
        print(f"  Variant {i} : file://{path.resolve()}")

if __name__ == "__main__":
    main()
