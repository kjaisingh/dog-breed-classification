#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:07:18 2019

@author: jaisi8631
"""

# unzip initial zip file
from zipfile import ZipFile

zip = ZipFile("stanford-dogs-dataset.zip")
zip.extractall("stanford-dogs-dataset")


# untar secondary tarball
import tarfile

fname = "stanford-dogs-dataset/images.tar"
tar = tarfile.open(fname, "r:")
tar.extractall("stanford-dogs-dataset")
tar.close()