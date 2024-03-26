import hashlib
import os
import requests 
import numpy as np
import gzip

def fetch(url):
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest() 
    fp = os.path.join("/tmp", url_hash)
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype= np.uint8).copy()

