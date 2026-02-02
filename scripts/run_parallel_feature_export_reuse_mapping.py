import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

import chromopho.mosaic as mosaic_mod
from chromopho.bipolar_image import BipolarImageProcessor
from chromopho.feature_extraction import extract_features
from chromopho.utils import save_structured_features



# create mosaic
SEED = 0
N_CELLS = 25000
SHAPE = "circle"

# set subtype params
ALPHA_CENTER = 1.0

ALPHA_SURROUND_S = 0.5
ALPHA_SURROUND_LM = 0.9
ALPHA_SURROUND_DIFF = 1.3

CENTER_SIGMA_S = 9.0
SURROUND_SIGMA_S = CENTER_SIGMA_S * 9
CENTER_SIGMA_LM = 7.0
SURROUND_SIGMA_LM = CENTER_SIGMA_LM * 9
CENTER_SIGMA_DIFF = 18.0
SURROUND_SIGMA_DIFF = CENTER_SIGMA_DIFF * 9

APPLY_RECTIFICATION = True
ON_K = 0.7
ON_N = 2.0
OFF_K = 0.7
OFF_N = 1.5

NONLIN_ADAPT_CONES = True


FEATURE_METHOD = "grayscale"  # passed into BipolarImageProcessor / extract_features

# parallel / runtime parameters
DEFAULT_WORKERS = 10
MAX_DEFAULT_WORKERS = None  # set to an int to cap auto-selection

# make default img, size should match img size that we extract bipolar mosaic responses from 
WHITE_IMG_SIZE = 400  # script assumes H=W=400 for mapping


def choose_default_workers(default=DEFAULT_WORKERS, max_cap=MAX_DEFAULT_WORKERS, leave_free=1):
    """Return number of worker processes to use.

    If default is a positive int, return it; otherwise use logical_cpu_count - leave_free,
    optionally capped by max_cap.
    """
    if isinstance(default, int) and default > 0:
        return default
    n = os.cpu_count() or 1
    n_use = max(1, n - (leave_free or 0))
    if isinstance(max_cap, int) and max_cap > 0:
        n_use = min(n_use, max_cap)
    return n_use




def build_mosaic():
    """Build the BipolarMosaic using the same configuration as run_parallel_feat_export."""
    seed = SEED
    np.random.seed(seed)

    alpha_center = ALPHA_CENTER

    alpha_surround_s = ALPHA_SURROUND_S
    alpha_surround_lm = ALPHA_SURROUND_LM
    alpha_surround_diff = ALPHA_SURROUND_DIFF

    center_sigma_s = CENTER_SIGMA_S
    surround_sigma_s = SURROUND_SIGMA_S

    center_sigma_lm = CENTER_SIGMA_LM
    surround_sigma_lm = SURROUND_SIGMA_LM

    center_sigma_diff = CENTER_SIGMA_DIFF
    surround_sigma_diff = SURROUND_SIGMA_DIFF

    nonlin_adapt_cones = NONLIN_ADAPT_CONES
    apply_rectification = APPLY_RECTIFICATION

    on_k = ON_K
    on_n = ON_N
    off_k = OFF_K
    off_n = OFF_N

    s_ratio = 3 / 100
    lm_ratio = 15 / 100
    dif_ratio = 15.5 / 100

    rec_frac = 0.15
    rf_size_s = 2 + int(np.floor(rec_frac * surround_sigma_s))
    rf_size_lm = int(np.floor(rec_frac * surround_sigma_lm))
    rf_size_diff = int(np.floor(rec_frac * surround_sigma_diff))

    s_rf_params = {
        "center_sigma": center_sigma_s,
        "surround_sigma": surround_sigma_s,
        "alpha_center": alpha_center,
        "alpha_surround": alpha_surround_s,
        "apply_rectification": apply_rectification,
        "on_k": on_k,
        "on_n": on_n,
        "off_k": off_k,
        "off_n": off_n,
        "nonlin_adapt_cones": nonlin_adapt_cones,
    }
    s_off = mosaic_mod.BipolarSubtype(
        name="s_off",
        ratio=s_ratio,
        rf_size=rf_size_s,
        color_filter_params={"center": "-s", "surround": "+lm"},
        rf_params=s_rf_params,
    )
    s_on = mosaic_mod.BipolarSubtype(
        name="s_on",
        ratio=s_ratio,
        rf_size=rf_size_s,
        color_filter_params={"center": "+s", "surround": "-lm"},
        rf_params=s_rf_params,
    )

    lm_rf_params = {
        "center_sigma": center_sigma_lm,
        "surround_sigma": surround_sigma_lm,
        "alpha_center": alpha_center,
        "alpha_surround": alpha_surround_lm,
        "apply_rectification": apply_rectification,
        "on_k": on_k,
        "on_n": on_n,
        "off_k": off_k,
        "off_n": off_n,
        "nonlin_adapt_cones": nonlin_adapt_cones,
    }
    m_on = mosaic_mod.BipolarSubtype(
        name="m_on",
        ratio=lm_ratio,
        rf_size=rf_size_lm,
        color_filter_params={"center": "+m", "surround": "-lm"},
        rf_params=lm_rf_params,
    )
    m_off = mosaic_mod.BipolarSubtype(
        name="m_off",
        ratio=lm_ratio,
        rf_size=rf_size_lm,
        color_filter_params={"center": "-m", "surround": "+lm"},
        rf_params=lm_rf_params,
    )
    l_on = mosaic_mod.BipolarSubtype(
        name="l_on",
        ratio=lm_ratio,
        rf_size=rf_size_lm,
        color_filter_params={"center": "+l", "surround": "-lm"},
        rf_params=lm_rf_params,
    )
    l_off = mosaic_mod.BipolarSubtype(
        name="l_off",
        ratio=lm_ratio,
        rf_size=rf_size_lm,
        color_filter_params={"center": "-l", "surround": "+lm"},
        rf_params=lm_rf_params,
    )

    dif_rf_params = {
        "center_sigma": center_sigma_diff,
        "surround_sigma": surround_sigma_diff,
        "alpha_center": alpha_center,
        "alpha_surround": alpha_surround_diff,
        "apply_rectification": apply_rectification,
        "on_k": on_k,
        "on_n": on_n,
        "off_k": off_k,
        "off_n": off_n,
        "nonlin_adapt_cones": nonlin_adapt_cones,
    }
    dif_on = mosaic_mod.BipolarSubtype(
        name="dif_on",
        ratio=dif_ratio,
        rf_size=rf_size_diff,
        color_filter_params={"center": "+lm", "surround": "-lm"},
        rf_params=dif_rf_params,
    )
    dif_off = mosaic_mod.BipolarSubtype(
        name="dif_off",
        ratio=dif_ratio,
        rf_size=rf_size_diff,
        color_filter_params={"center": "-lm", "surround": "+lm"},
        rf_params=dif_rf_params,
    )

    mosaic = mosaic_mod.BipolarMosaic(
        N_CELLS,
        shape=SHAPE,
        subtypes=[m_off, m_on, l_off, l_on, s_off, s_on, dif_off, dif_on],
    )
    return mosaic




_WORK_MOSAIC = None
_WORK_BIP = None
_WORK_WHITE_SHAPE = None


def init_worker(mosaic, white_shape):
    """Initializer for each worker process: build mosaic + mapping from white image.

    Each worker constructs a BipolarImageProcessor once using a 400x400 all-white
    RGB image to define the mosaic->pixel mapping. Subsequent images reuse that
    mapping via process_new_image.
    """
    global _WORK_MOSAIC, _WORK_BIP, _WORK_WHITE_SHAPE
    _WORK_MOSAIC = mosaic
    _WORK_WHITE_SHAPE = tuple(white_shape)

    h, w = _WORK_WHITE_SHAPE
    white_img = np.ones((h, w, 3), dtype=float)

    # Build the processor once; this creates the receptive field map.
    _WORK_BIP = BipolarImageProcessor(_WORK_MOSAIC, white_img, save_flat=False)


def process_one_image(img_path, output_dir, done_dir, verbose):
    """Process a single image using worker-global BipolarImageProcessor mapping."""
    global _WORK_BIP, _WORK_WHITE_SHAPE

    img = plt.imread(img_path)


    h_expected, w_expected = _WORK_WHITE_SHAPE
    if img.shape[0] != h_expected or img.shape[1] != w_expected:
        raise ValueError(
            f"Image {img_path} has shape {img.shape[:2]}, expected {(h_expected, w_expected)}.
            Failed to extract features (!!!!!)"
        )
        return # 

    # Reuse mapping: update image and recompute bipolar responses + per-pixel map.
    _WORK_BIP.process_new_image(
        img,
        method=FEATURE_METHOD,
        save_flat=False,
        stimulation_mosaic=None,
        amacrine_sigma_blur=None,
        recompute_pixel_map=True,
    )

    # extract per-pixel features using existing helper.
    features_array, labels_array = extract_features(_WORK_BIP, return_labels=True)

    # save results 
    filename_base = os.path.splitext(os.path.basename(img_path))[0]
    save_structured_features(features_array, output_dir, filename_base, "_features.npy")
    save_structured_features(labels_array, output_dir, filename_base, "_labels.npy")

    # Move image to done_dir 
    try:
        os.replace(img_path, os.path.join(done_dir, os.path.basename(img_path)))
    except Exception:
        if verbose:
            print(f"Warning: could not move {img_path} to {done_dir}")

    if verbose:
        print(f"Saved features for {img_path}")

    return filename_base





def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel feature extraction using a BipolarMosaic with a fixed "
            "400x400 pixel mapping (built from an all-white image)."
        )
    )
    parser.add_argument("--image_dir", required=True, help="Directory of input .png images (400x400).")
    parser.add_argument("--output_dir", required=True, help="Directory to save feature arrays.")
    parser.add_argument("--done_dir", required=True, help="Directory to move processed images into.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.done_dir, exist_ok=True)

    mosaic = build_mosaic()

    img_paths = [
        os.path.join(args.image_dir, fn)
        for fn in os.listdir(args.image_dir)
        if fn.lower().endswith(".png")
    ]
    if args.verbose:
        print(f"Found {len(img_paths)} images")

    workers = args.workers if (args.workers and args.workers > 0) else choose_default_workers()

    white_shape = (WHITE_IMG_SIZE, WHITE_IMG_SIZE)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(mosaic, white_shape),
    ) as ex:
        futures = {
            ex.submit(process_one_image, p, args.output_dir, args.done_dir, args.verbose): p
            for p in img_paths
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                _ = fut.result()
                if args.verbose:
                    print(f"Done {p}")
            except Exception as e:
                print(f"Error processing {p}: {e}")


if __name__ == "__main__":
    main()
