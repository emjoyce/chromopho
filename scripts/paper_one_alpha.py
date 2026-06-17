import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

import chromopho.mosaic as mosaic_mod
from chromopho.bipolar_image_tf import BipolarImageProcessorTF




# create mosaic
SEED = 0
N_CELLS = 25000
SHAPE = "circle"

# set subtype params
ALPHA_CENTER = 1.0

# TODO: these will be sampled from dists for m and d
ALPHA_SURROUND_S = 0.5
ALPHA_SURROUND_LM = 0.9
ALPHA_SURROUND_DIFF = 0.9
# keep only the number of surround-alpha bins here.
# the subtype now derives the actual alpha values from its family.
# SURROUND_ALPHA_N_BINS = 5

# DONE: decide vals
# dendritic radius in microns plus 
CENTER_SIGMA_S = 3.9#9.0
SURROUND_SIGMA_S = CENTER_SIGMA_S * 9
CENTER_SIGMA_LM = 2.2 #7.0
SURROUND_SIGMA_LM = CENTER_SIGMA_LM * 9
CENTER_SIGMA_DIFF = 6.2 #18.0
SURROUND_SIGMA_DIFF = CENTER_SIGMA_DIFF * 9

# DONE: new vals
APPLY_RECTIFICATION = True
ON_K = 0.5
ON_N = 2.5
OFF_K = 0.5
OFF_N = 4.0

NONLIN_ADAPT_CONES = True

FEATURE_METHOD = "grayscale"
DEFAULT_WORKERS = 10
MAX_DEFAULT_WORKERS = None
WHITE_IMG_SIZE = 400



# output params
SAVE_DTYPE = "float32"  # dtype for saved .npy arrays

# optional amacrine blur applied to bipolar cell outputs
# blur step happens after, when I train the NN on the saved features, 
    # so that I can experiment with different blur levels without re-extracting feats
    # also so I can place phosphene and just blur once 
AMACRINE_SIGMA_BLUR = None



# DEFAULT_IMAGE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/to_do"
# DEFAULT_OUTPUT_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/paper_feats"
# DEFAULT_DONE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/done"
DEFAULT_IMAGE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done_2"
DEFAULT_OUTPUT_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/paper_s_feats"
DEFAULT_DONE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done"



def choose_default_workers(default=DEFAULT_WORKERS, max_cap=MAX_DEFAULT_WORKERS, leave_free=1):
    """
    Return number of worker processes to use:
    - If default is a positive int, return it.
    - Otherwise use (logical_cpu_count - leave_free), optionally capped by max_cap.
    """
    if isinstance(default, int) and default > 0:
        return default
    n = os.cpu_count() or 1
    n_use = max(1, n - (leave_free or 0))
    if isinstance(max_cap, int) and max_cap > 0:
        n_use = min(n_use, max_cap)
    return n_use

def build_mosaic(seed=SEED):
    """Build the BipolarMosaic using the same configuration as run_parallel_feat_export."""
    np.random.seed(seed)

    alpha_center = ALPHA_CENTER

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

    alpha_surround_diff = ALPHA_SURROUND_DIFF
    alpha_surround_lm = ALPHA_SURROUND_LM
    alpha_surround_s = ALPHA_SURROUND_S

    # s_ratio = 3 / 100
    # lm_ratio = 15 / 100
    # dif_ratio = 15.5 / 100
    # NEW
    s_ratio = 3/100 # split into 2 subtypes
    lm_ratio = 13.5/100 # split into 4 subtypes
    dif_ratio = 20/100 # split into 2 subtypes

    # surround_alpha_n_bins = SURROUND_ALPHA_N_BINS

    rec_frac = 0.15
    rf_size_s = 2 + int(np.floor(rec_frac * surround_sigma_s))
    rf_size_lm = int(np.floor(rec_frac * surround_sigma_lm))
    rf_size_diff = int(np.floor(rec_frac * surround_sigma_diff))

    s_rf_params = {
        "family": None, #"midget",
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
        # "surround_alpha_n_bins": surround_alpha_n_bins,
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
        "family": None,#"midget",
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
        #"surround_alpha_n_bins": surround_alpha_n_bins,
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
        "family": None, #"diffuse",
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
        #"surround_alpha_n_bins": surround_alpha_n_bins,
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

# Worker-global state (set inside init_worker in each process)
_WORK_MOSAIC = None
_WORK_BIP = None
_WORK_WHITE_SHAPE = (WHITE_IMG_SIZE, WHITE_IMG_SIZE)

def init_worker(mosaic):
    """Initializer for each worker process: build mapping from a white image."""
    global _WORK_MOSAIC, _WORK_BIP, _WORK_WHITE_SHAPE

    _WORK_MOSAIC = mosaic

    h, w = _WORK_WHITE_SHAPE
    white_img = np.ones((h, w, 3), dtype=float)

    # Use TF-based processor here
    _WORK_BIP = BipolarImageProcessorTF(_WORK_MOSAIC, white_img)

def process_one_image(img_path, output_dir, done_dir, verbose):
    """Process a single image using worker-global BipolarImageProcessor mapping."""
    global _WORK_BIP, _WORK_WHITE_SHAPE

    img = plt.imread(img_path)


    h_expected, w_expected = _WORK_WHITE_SHAPE
    if img.shape[0] != h_expected or img.shape[1] != w_expected:
        raise ValueError(
            f"Image {img_path} has shape {img.shape[:2]}, expected {(h_expected, w_expected)}.Failed to extract features (!!!!!)"
        )
        return # 

    # Reuse mapping: update image and recompute bipolar responses + per-pixel map.
    _WORK_BIP.process_new_image(
        img,
        method=FEATURE_METHOD,
        stimulation_mosaic=None,
        amacrine_sigma_blur=None,
        recompute_pixel_map=True,
    )


    grid_outputs = np.asarray(_WORK_BIP.grid_outputs)

    # save results 
    filename_base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_dir, filename_base + '_mosaic_output.npy')
    np.save(out_path, grid_outputs.astype(SAVE_DTYPE))
    

    # move processed image to done_dir (best-effort)
    
    try:
        os.replace(img_path, os.path.join(done_dir, os.path.basename(img_path)))
    except Exception:
        if verbose:
            print(f"Warning: could not move {img_path} to {done_dir}")

    return out_path

def main():
    parser = argparse.ArgumentParser(description="Parallel export mosaic outputs")
    parser.add_argument('--image_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--done_dir', default=DEFAULT_DONE_DIR)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--amacrine_sigma_blur', type=float, default=AMACRINE_SIGMA_BLUR,
                        help='Optional masked amacrine blur applied to output of bipolar cells (None disables)')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.done_dir, exist_ok=True)

    mosaic = build_mosaic(seed=args.seed)

    img_paths = [os.path.join(args.image_dir, fn) for fn in os.listdir(args.image_dir) if fn.lower().endswith('.png')]
    if args.verbose:
        print(f"Found {len(img_paths)} images")

    workers = args.workers if (args.workers and args.workers > 0) else choose_default_workers()

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(mosaic,)) as ex:
        futures = {ex.submit(process_one_image, p, args.output_dir, args.done_dir, args.verbose): p for p in img_paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                out = fut.result()
                if args.verbose:
                    print(f"Done {p} -> {out}")
            except Exception as e:
                print(f"Error processing {p}: {e}")

if __name__ == '__main__':
    main()