import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt

import chromopho.mosaic as mosaic_mod
from chromopho.bipolar_image import BipolarImageProcessor

# -------------------------
# CONFIGURATION (edit here)
# grouped by the function/feature they configure
# -------------------------

# --- Mosaic / subtype parameters (used by build_mosaic) ---
SEED = 10
N_CELLS = 25000
SHAPE = "circle"

# receptive field / subtype tuning
ALPHA_CENTER = 1.0
ALPHA_SURROUND_S = 0.25
ALPHA_SURROUND_LM = 0.65
ALPHA_SURROUND_DIFF = 0.45

CENTER_SIGMA_S = 8.0
SURROUND_SIGMA_S = 25.0
CENTER_SIGMA_LM = 3.0
SURROUND_SIGMA_LM = 10.0
CENTER_SIGMA_DIFF = 6.0
SURROUND_SIGMA_DIFF = 8.0

REC_KIND = "softplus"
REC_R0 = 0.1
REC_ALPHA = 0.05
REC_BETA = 5

# --- Processing / output parameters (used by process_one_image) ---
DEFAULT_GAUSSIAN_BLUR = False
DEFAULT_GAUSSIAN_SIGMA = 1.0
SAVE_DTYPE = "float32"  # dtype for saved .npy arrays

# --- Parallel / runtime parameters (used by main) ---
DEFAULT_WORKERS = 10   # explicit CLI override if set # conservative - i have 12 performance cores
MAX_DEFAULT_WORKERS = None  # set to an int to cap auto-selection (e.g. 12)

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

# --- I/O defaults (can be overridden by CLI) ---
DEFAULT_IMAGE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/to_do_natural_400'
DEFAULT_OUTPUT_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/grid_output_old_params' 
DEFAULT_DONE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done_2'

# test
# DEFAULT_IMAGE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/to_do'
# DEFAULT_OUTPUT_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/grid_output_old_params' 
# DEFAULT_DONE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/done'

# -------------------------
# end CONFIGURATION
# -------------------------

# Worker-global mosaic (set in initializer)
_WORK_MOSAIC = None

def init_worker(m):
    global _WORK_MOSAIC
    _WORK_MOSAIC = m

def process_one_image(img_path, output_dir, analysis_complete_dir, gaussian_blur, gaussian_sigma, verbose):
    try:
        img = plt.imread(img_path)
    except Exception:
        from imageio import imread
        img = imread(img_path)

    # create BipolarImageProcessor and compute per-cell flattened outputs
    bip = BipolarImageProcessor(_WORK_MOSAIC, img, save_flat = True)
    grid_outputs = bip.grid_outputs
    if gaussian_blur is not False:
        grid_outputs = gaussian_filter(grid_outputs, sigma=gaussian_sigma)

    # save results
    filename_base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_dir, filename_base + '_mosaic_output.npy')
    np.save(out_path, grid_outputs.astype(SAVE_DTYPE))

    # move processed image to analysis_complete_dir (best-effort)
    try:
        os.replace(img_path, os.path.join(analysis_complete_dir, os.path.basename(img_path)))
    except Exception:
        if verbose:
            print(f"Warning: could not move {img_path} to {analysis_complete_dir}")

    if verbose:
        print(f"Saved {out_path}")
    return out_path

def build_mosaic():
    # uses CONFIG variables above
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

    rec_kind = REC_KIND
    rec_r0 = REC_R0
    rec_alpha = REC_ALPHA
    rec_beta = REC_BETA

    s_rf_params = {'center_sigma':center_sigma_s, 'surround_sigma':surround_sigma_s, 'alpha_center':1, 'alpha_surround':alpha_surround_s, 'rec_kind':rec_kind, 'rec_beta':rec_beta}
    s_off = mosaic_mod.BipolarSubtype(name = 's_off', ratio = 2.5/100,  rf_size = 15, color_filter_params = {'center':'-s', 'surround':'+lm'}, rf_params=s_rf_params)
    s_on  = mosaic_mod.BipolarSubtype(name = 's_on',  ratio = 2.5/100,  rf_size = 15, color_filter_params = {'center':'+s', 'surround':'-lm'}, rf_params=s_rf_params)

    lm_rf_params = {'center_sigma':center_sigma_lm, 'surround_sigma':surround_sigma_lm, 'alpha_center':1, 'alpha_surround':alpha_surround_lm, 'rec_kind':rec_kind, 'rec_beta':rec_beta}
    m_on  = mosaic_mod.BipolarSubtype(name = 'm_on',  ratio = 18/100, rf_size = 6,  color_filter_params = {'center':'+m', 'surround':'-l'}, rf_params=lm_rf_params)
    m_off = mosaic_mod.BipolarSubtype(name = 'm_off', ratio = 18/100, rf_size = 6,  color_filter_params = {'center':'-m', 'surround':'+l'}, rf_params=lm_rf_params)

    l_on  = mosaic_mod.BipolarSubtype(name = 'l_on',  ratio = 18/100, rf_size = 6,  color_filter_params = {'center':'+l', 'surround':'-m'}, rf_params=lm_rf_params)
    l_off = mosaic_mod.BipolarSubtype(name = 'l_off', ratio = 19/100, rf_size = 6,  color_filter_params = {'center':'-l', 'surround':'+m'}, rf_params=lm_rf_params)

    dif_rf_params = {'center_sigma':center_sigma_diff, 'surround_sigma':surround_sigma_diff, 'alpha_center':1, 'alpha_surround':alpha_surround_diff, 'rec_kind':rec_kind, 'rec_beta': rec_beta}
    dif_on  = mosaic_mod.BipolarSubtype(name = 'dif_on',  ratio = 10/100, rf_size = 8, color_filter_params = {'center':'+lm', 'surround':'-lm'})
    dif_off = mosaic_mod.BipolarSubtype(name = 'dif_off', ratio = 10/100, rf_size = 8, color_filter_params = {'center':'-lm', 'surround':'+lm'})

    mosaic = mosaic_mod.BipolarMosaic(N_CELLS, shape = SHAPE, subtypes = [m_off, m_on, l_off, l_on, s_off, s_on, dif_on, dif_off])
    return mosaic

def main():
    parser = argparse.ArgumentParser(description="Parallel export mosaic outputs")
    parser.add_argument('--image_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--done_dir', default=DEFAULT_DONE_DIR)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--gaussian_blur', action='store_true', default=DEFAULT_GAUSSIAN_BLUR)
    parser.add_argument('--gaussian_sigma', type=float, default=DEFAULT_GAUSSIAN_SIGMA)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.done_dir, exist_ok=True)

    mosaic = build_mosaic()

    img_paths = [os.path.join(args.image_dir, fn) for fn in os.listdir(args.image_dir) if fn.lower().endswith('.png')]
    if args.verbose:
        print(f"Found {len(img_paths)} images")

    workers = args.workers if (args.workers and args.workers > 0) else choose_default_workers()

    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(mosaic,)) as ex:
        futures = {ex.submit(process_one_image, p, args.output_dir, args.done_dir, args.gaussian_blur, args.gaussian_sigma, args.verbose): p for p in img_paths}
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