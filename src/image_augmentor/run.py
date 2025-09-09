"""
CLI entry for TFT augmentation. Run from with_label.
"""

from params import build_arg_parser, load_params_from_args, make_rng
from dataset import list_sample_folders
from runner import process_sample
from colorama import init, Fore, Back, Style

def main():
    init(autoreset=True)

    parser = build_arg_parser()
    args = parser.parse_args()

    params = load_params_from_args(args)
    rng = make_rng(params["seed"])

    root = args.root
    total_aug = 0
    samples_with_output = 0

    for sample_dir in list_sample_folders(root):
        # Copy per-sample so max dims can be resolved per image
        tmp_params = params.copy()
        gen = process_sample(sample_dir, tmp_params, rng)
        total_aug += gen
        if gen > 0:
            samples_with_output += 1

    TOTAL = Back.YELLOW + "[TOTAL]" + Style.RESET_ALL
    print(f"{TOTAL} Generated {total_aug} augmented set(s) across {samples_with_output} sample folder(s).")

if __name__ == "__main__":
    main()