#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 18:52:33 2024

@author: fvillalobos
"""

import sys
from os.path import exists, basename, join, exists
import numpy as np
from misc import loadimage, isimage, processfolder
from asengine import asengine

def checkimage(img):
    return img is not None and img.size > 0 and img.ndim == 3 and img.dtype == np.uint8

def processfile(model, filein, fileout):
    # fileout is ignored, it's just kept for compatibility with processfolder
    ASE = model
    img = loadimage(filein)
    score, debug = ASE.processL1(img)
    ASE.printCSVline(basename(filein), score, debug)
    return True

def main():
    if len(sys.argv) != 3:
        print("Args: AS model folder, input file/folder")
    else:
        fin = sys.argv[2]
        assert exists(fin)
        ASE = asengine(sys.argv[1])
        assert ASE.ready
        ASE.printCSVheader()
        if isimage(fin):
            processfile(ASE, fin, None)
        else:
            processfolder(ASE, fin, None, processfile, verbose=False)

if __name__ == "__main__":
    main()