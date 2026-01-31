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
SEED = 0
N_CELLS = 25000
SHAPE = "circle"

# receptive field / subtype tuning
ALPHA_CENTER = 1.0

ALPHA_SURROUND_S = 0.5
ALPHA_SURROUND_LM = 0.9
ALPHA_SURROUND_DIFF = 1.3

CENTER_SIGMA_S = 9.0
SURROUND_SIGMA_S = CENTER_SIGMA_S*9
CENTER_SIGMA_LM = 7.0
SURROUND_SIGMA_LM = CENTER_SIGMA_LM*9
CENTER_SIGMA_DIFF = 18.0
SURROUND_SIGMA_DIFF = CENTER_SIGMA_DIFF*9

APPLY_RECTIFICATION = True
ON_K = 0.7
ON_N = 2.0
OFF_K = 0.7
OFF_N = 1.5

NONLIN_ADAPT_CONES = True



# --- Processing / output parameters (used by process_one_image) ---
SAVE_DTYPE = "float32"  # dtype for saved .npy arrays

# optional amacrine blur applied to bipolar cell outputs (None -> disabled)
AMACRINE_SIGMA_BLUR = None

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
# DEFAULT_IMAGE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/to_do_natural_400'
# DEFAULT_OUTPUT_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/grid_output_old_params' 
# DEFAULT_DONE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done_2'

# test
# DEFAULT_IMAGE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/to_do'
# DEFAULT_OUTPUT_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/grid_output_old_params' 
# DEFAULT_DONE_DIR = '/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/done'

# sfn mini
# DEFAULT_IMAGE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/sfn_mini/to_do"
# DEFAULT_OUTPUT_DIR = "//Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/sfn_mini/feats"
# DEFAULT_DONE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/sfn_mini/done"

# sfn mini test images
# DEFAULT_IMAGE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/sfn_mini/to_do"
# DEFAULT_OUTPUT_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/sfn_mini/feats"
# DEFAULT_DONE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/test_imgs/sfn_mini/done"

# sfn full 
# DEFAULT_IMAGE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done"
# DEFAULT_OUTPUT_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/sfn_full_feats"
# DEFAULT_DONE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done_2"

# solid imgs sfn
DEFAULT_IMAGE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/solid_color_manifold_imgs"
DEFAULT_OUTPUT_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/sfn_manifold_feats"
DEFAULT_DONE_DIR = "/Users/emilyjoyce/repos/chromopho/chromopho/imgs/done"

# -------------------------
# end CONFIGURATION
# -------------------------

# Worker-global mosaic (set in initializer)
_WORK_MOSAIC = None

def init_worker(m):
    global _WORK_MOSAIC
    _WORK_MOSAIC = m

def process_one_image(img_path, output_dir, analysis_complete_dir, amacrine_sigma_blur, verbose):
    try:
        img = plt.imread(img_path)
    except Exception:
        from imageio import imread
        img = imread(img_path)

    # create BipolarImageProcessor and compute per-cell flattened outputs
    # pass optional amacrine blur into the processor (None disables it)
    bip = BipolarImageProcessor(_WORK_MOSAIC, img, save_flat = True, amacrine_sigma_blur=amacrine_sigma_blur)
    grid_outputs = bip.grid_outputs

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

    nonlin_adapt_cones = NONLIN_ADAPT_CONES
    apply_rectification = APPLY_RECTIFICATION

    on_k = ON_K
    on_n = ON_N
    off_k = OFF_K
    off_n = OFF_N

    s_ratio = 3/100
    lm_ratio = 15/100
    dif_ratio = 15.5/100

    rec_frac = .15
    rf_size_s = 2+int(np.floor(rec_frac*surround_sigma_s))
    rf_size_lm = int(np.floor(rec_frac*surround_sigma_lm))
    rf_size_diff = int(np.floor(rec_frac*surround_sigma_diff))



    s_rf_params = {'center_sigma':center_sigma_s, 'surround_sigma':surround_sigma_s, 'alpha_center':alpha_center, 
                            'alpha_surround':alpha_surround_s,  'apply_rectification':apply_rectification, 
                            'on_k':on_k, 'on_n':on_n, 'off_k':off_k, 'off_n':off_n, 'nonlin_adapt_cones': nonlin_adapt_cones}
    s_off = mosaic_mod.BipolarSubtype(name = 's_off', ratio = s_ratio,  rf_size = rf_size_s, color_filter_params = {'center':'-s', 'surround':'+lm'}, rf_params=s_rf_params)
    s_on  = mosaic_mod.BipolarSubtype(name = 's_on',  ratio = s_ratio,  rf_size = rf_size_s, color_filter_params = {'center':'+s', 'surround':'-lm'}, rf_params=s_rf_params)


    lm_rf_params = {'center_sigma':center_sigma_lm, 'surround_sigma':surround_sigma_lm, 'alpha_center':alpha_center, 
                    'alpha_surround':alpha_surround_lm, 'apply_rectification':apply_rectification, 
                            'on_k':on_k, 'on_n':on_n, 'off_k':off_k, 'off_n':off_n, 'nonlin_adapt_cones': nonlin_adapt_cones}
    m_on  = mosaic_mod.BipolarSubtype(name = 'm_on',  ratio = lm_ratio, rf_size = rf_size_lm,  color_filter_params = {'center':'+m', 'surround':'-lm'}, rf_params=lm_rf_params)
    m_off = mosaic_mod.BipolarSubtype(name = 'm_off', ratio = lm_ratio, rf_size = rf_size_lm,  color_filter_params = {'center':'-m', 'surround':'+lm'}, rf_params=lm_rf_params)
    l_on  = mosaic_mod.BipolarSubtype(name = 'l_on',  ratio = lm_ratio, rf_size = rf_size_lm,  color_filter_params = {'center':'+l', 'surround':'-lm'}, rf_params=lm_rf_params)
    l_off = mosaic_mod.BipolarSubtype(name = 'l_off', ratio = lm_ratio, rf_size = rf_size_lm,  color_filter_params = {'center':'-l', 'surround':'+lm'}, rf_params=lm_rf_params)

    dif_rf_params = {'center_sigma':center_sigma_diff, 'surround_sigma':surround_sigma_diff, 'alpha_center':alpha_center, 
                        'alpha_surround':alpha_surround_diff, 'apply_rectification':apply_rectification, 
                            'on_k':on_k, 'on_n':on_n, 'off_k':off_k, 'off_n':off_n, 'nonlin_adapt_cones': nonlin_adapt_cones}

    dif_on  = mosaic_mod.BipolarSubtype(name = 'dif_on',  ratio = dif_ratio, rf_size = rf_size_diff, color_filter_params = {'center':'+lm', 'surround':'-lm'}, rf_params=dif_rf_params)
    dif_off = mosaic_mod.BipolarSubtype(name = 'dif_off', ratio = dif_ratio, rf_size = rf_size_diff, color_filter_params = {'center':'-lm', 'surround':'+lm'}, rf_params=dif_rf_params)

    mosaic = mosaic_mod.BipolarMosaic(N_CELLS, shape = SHAPE, subtypes = [m_off, m_on, l_off, l_on, s_off, s_on, dif_off, dif_on])
    return mosaic

def main():
    parser = argparse.ArgumentParser(description="Parallel export mosaic outputs")
    parser.add_argument('--image_dir', default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--done_dir', default=DEFAULT_DONE_DIR)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--amacrine_sigma_blur', type=float, default=AMACRINE_SIGMA_BLUR,
                        help='Optional masked amacrine blur applied to output of bipolar cells (None disables)')
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
        futures = {ex.submit(process_one_image, p, args.output_dir, args.done_dir, args.amacrine_sigma_blur, args.verbose): p for p in img_paths}
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