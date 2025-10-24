# params.py
# Central config for the repair_gen pipeline.

class Params:
    # Path to dataset root when running run.py from the dataset root
    dataset_root = "."

    # Which stages to run
    steps = {
        "augment": True,        # random_negative_augmentor.py
        "visualize": True,      # visualize_random_overlays.py
        "extract_rules": True,  # repair_rule_extractor.py
    }

    # -------- Augmentor --------
    augment = {
        "n": 130,                        # how many negatives to generate
        "include": ["10", "11", "110", "112"],  # rps types in the randomizer pool
        "ops": ["shift", "delete"],     # any of: shift, scale, delete
        "shift_std": 15.0,              # pixels (Gaussian std)
        "scale_std": 0.40,              # unitless (Gaussian std around 1.0)
        "p_delete": 0.15,               # probability to delete a touched contour
        "img_w": 512,                   # fallback width
        "img_h": 512,                   # fallback height
        "seed": None,                   # set to an int for reproducibility
    }

    # -------- Visualizer --------
    visualize = {
        "defect_open": False,           # draw defect polygons closed by default
        "verbose": True,
    }

    # -------- Repair rule extractor --------
    extract_rules = {
        "only_missing": False,          # set True to skip existing repair_rule.json
        "mask_thresh": 128,             # binarization threshold for masks/*.jpg
        "min_pixels": 1,                # min intersection pixels (with the defect region) to count as present
        "verbose": True,
    }
