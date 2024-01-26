#! /usr/bin/env python

try:
    import OMETiffTools
    import argparse

except ImportError:
    raise ImportError("Some modules could not be imported.")

def main():
    parser=argparse.ArgumentParser(description="Merge multi-file OME-Tiff into OME BigTiff file")

    parser.add_argument("masterfile", help="Path to master Tiff file with full OME-XML header")

    parser.add_argument("-p", "--progress", 
                        choices=["tqdm", "plain", "none"],
                        default = "tqdm",
                        help="type of progress meter (tqdm, plain, none)"
                        )

    parser.add_argument("-o", "--output_file", 
                        required=True,
                        help="name and path of output BigTiff file."
                        )


    args=parser.parse_args()


    otc=OMETiffTools.OMETiffConcat(args.masterfile)
    print(f"Number of data files: {otc.nfiles}\n")

    otc.write_bigtiff(args.output_file, progress=args.progress)

if __name__ == "__main__":
    main()
