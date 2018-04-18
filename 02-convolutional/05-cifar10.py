import os
import gzip
import urllib.request
import numpy as np
import tensorflow as tf

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def download_from_url(url, filename, folder='./', notifications = True):
    #testfile = urllib.URLopener()
    #testfile.retrieve(url, folder + filename)
    try:
        urllib.request.urlretrieve(url, folder + filename)
        if notifications:
            print('[OK] {filename} downloaded succesfully from {url}'.format(
                filename=filename, url=url
            ))
    except:
        print('[Error] Error while getting {}'.format(url))
def uncompress_gzip(file):
    f = gzip.open(file, 'rb')
    file_content = f.read()
    f.close()

create_folder('./samples')
download_from_url(url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                filename = 'cifar-10.tar.gz', folder='./samples/')
