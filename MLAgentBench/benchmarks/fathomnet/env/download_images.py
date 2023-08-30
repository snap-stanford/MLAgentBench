# -*- coding: utf-8 -*-
"""
download_images

Script to retrieve images for the 2023 FathomNet out-of-sample challenge as part of FGVC 10. 

Assumes COCO formated annotation file has been download from http://www.kaggle.com/competitions/fathomnet-out-of-sample-detection
"""
# Author: Eric Orenstein (eorenstein@mbari.org)

import os
import sys
import glob
import json
import requests
import logging
import argparse
import progressbar
import pandas as pd
from shutil import copyfileobj


def download_imgs(imgs, outdir=None):
    """
    Download images to an output dir
    
    :param imgs: list of urls 
    :param outdir: desired directory [default to working directory]
    :return :
    """

    # set the out directory to default if not specified
    if not outdir:
        outdir = os.path.join(os.getcwd(), 'images')

    # make the directory if it does not exist
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        logging.info(f"Created directory {outdir}")

    flag = 0  # keep track of how many image downloaded

    for name, url in progressbar.progressbar(imgs):
        file_name = os.path.join(
            outdir, name
        )

        # only download if the image does not exist in the outdir
        if not os.path.exists(file_name):
            resp = requests.get(url, stream=True)
            resp.raw.decode_content = True
            with open(file_name, 'wb') as f:
                copyfileobj(resp.raw, f)
            flag += 1

    logging.info(f"Downloaded {flag} new images to {outdir}")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Download images from a COCO annotation file")
    parser.add_argument('dataset', type=str, help='Path to json COCO annotation file')
    parser.add_argument('--outpath', type=str, default=None, help='Path to desired output folder')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    logging.info(f'opening {args.dataset}')
    with open(args.dataset, 'r') as ff:
        dataset = json.load(ff)

    ims = pd.DataFrame(dataset['images'])

    logging.info(f'retrieving {ims.shape[0]} images')

    ims = zip(ims['file_name'].to_list(), ims['coco_url'].to_list())

    # download images
    download_imgs(ims, outdir=args.outpath)
