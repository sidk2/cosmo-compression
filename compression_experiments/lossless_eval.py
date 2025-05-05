from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from cosmo_compression.data import data
from cosmo_compression.compressor.compressors import FPZIP, GZIP_SHUFFLE, LZMA

parser = ArgumentParser()
parser.add_argument(
    "--image_count",
    default=15000,
    help="15000 or smaller for debugging",
    type=int,
    required=False,
)

parser.add_argument(
    "--suite",
    default='Astrid',
    help="Astrid, or IllustrisTNG",
    type=str,
    required=False,
)

parser.add_argument(
    "--dataset",
    default='LH',
    help="LH, or WDM",
    type=str,
    required=False,
)

compressors_dict = {'gzip_shuffle': GZIP_SHUFFLE, 'fpzip': FPZIP, 'lzma': LZMA}

def main(args):
    map_type = 'Mcdm'
    redshift = 0.0
    all_processed_data = data.CAMELS(
        idx_list=range(args.image_count),
        suite=args.suite,
        map_type=map_type,
        dataset=args.dataset,
        parameters=['Omega_m', 'sigma_8',],
    )

    all_raw_data = np.load(
        Path('/monolith/global_data/astro_compression/CAMELS') / f"Maps_{map_type}_{args.suite}_{args.dataset}_z={redshift:.2f}.npy"
    )
    # processed data
    image_count = np.min([args.image_count, len(all_processed_data)]) # in case args.image_count is overwritten during data construction

    evaluated_methods = ['gzip_shuffle', 'fpzip', 'lzma']
    results = {}

    for key, values in vars(args).items():
        results[key] = values
    
    for method in evaluated_methods:
        # varies with method by looking up in compressors_dict
        CompressorMethod = compressors_dict[method]
        processed_bpp = np.zeros((1, image_count))
        pbar = tqdm(iterable=range(image_count), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
        for i in pbar:
            single_frame, _ = all_processed_data[i]
            compressor = CompressorMethod(single_frame)
            compressor.compress()
            processed_bpp[0, i] = compressor.get_bpp()
            compressor.decompress
        print(np.mean(processed_bpp))

        # raw data
        raw_bpp = np.zeros((1, image_count))
        pbar = tqdm(iterable=range(image_count), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')
        for i in pbar:
            single_frame = all_raw_data[i, :, :]
            compressor = CompressorMethod(single_frame)
            compressor.compress()
            raw_bpp[0, i] = compressor.get_bpp()
            compressor.decompress
        print(np.mean(raw_bpp))
    
        results[method] = {'processed_bpp': np.mean(processed_bpp), 'raw_bpp': np.mean(raw_bpp)}
        print(results)

    with open('lossless_eval_results.json', 'w') as f:
        json.dump(results, f)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)