"""
Extract individual wav files for Buckeye.

Author: Herman Kamper
Original Repository: https://github.com/kamperh/zerospeech2021_baseline/blob/2f2c47766ffc02574dcc71fea7fe5247ca4f323c/get_buckeye_wavs.py
Contact: kamperh@gmail.com
Date: 2021
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import librosa
import soundfile as sf
import sys
import os

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "buckeye_dir", type=str, help="local copy of the official Buckeye data"
        )
    parser.add_argument( # to have shorter segments for the model
        "--segments", action="store_true",
        help="read Buckeye segments instead of the full utterances"
        )
    parser.add_argument(
        "--felix", action="store_true",
        help="use the Felix Kreuk Buckeye splits"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    in_dir = Path(args.buckeye_dir)
    out_dir = os.path.split(in_dir)[0]
    
    if args.segments:
        out_dir = out_dir / Path("buckeye_segments/")
    elif args.felix:
        out_dir = out_dir / Path("buckeye_felix/")
    else:
        out_dir = out_dir / Path("buckeye/")

    # for split in ["train", "test", "val"]:
    for split in ["val", "test"]:
    # for split in ["test"]:
        print("Extracting utterances for {} set".format(split))

        split_path = out_dir / split
        
        if not split_path.with_suffix(".json").exists():
            print("Skipping {} (no json file)".format(split))
            continue
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in tqdm(metadata):
                wav_path = in_dir/in_path

                assert wav_path.with_suffix(".wav").exists(), (
                    "'{}' does not exist".format(
                    wav_path.with_suffix(".wav"))
                    )
                if args.segments:
                    out_path = os.path.split(in_dir)[0] / Path("buckeye_segments/")/split/Path(
                        out_path
                        ).stem
                elif args.felix:
                    out_path = os.path.split(in_dir)[0] / Path("buckeye_felix/")/split/Path(
                        out_path
                        ).stem
                else:
                    out_path = os.path.split(in_dir)[0] / Path("buckeye/")/split/Path(out_path).stem
                
                out_path.parent.mkdir(parents=True, exist_ok=True)
                wav, _ = librosa.load(
                    wav_path.with_suffix(".wav"), sr=16000,
                    offset=start, duration=duration
                    )
                sf.write(
                    out_path.with_suffix(".wav"), wav, samplerate=16000)

if __name__ == "__main__": # python3 wordseg/preprocess_buckeye.py --segments /media/hdd/data/buckeye/
    main()