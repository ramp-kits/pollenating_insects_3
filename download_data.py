"""Download data after cloning.

Run this once. It will create `data/imgs` and download the images there
using the names `<id>`, where `<id>`s are coming from `data/train.csv` and
`data/test.csv`. If images are properly unzipped in `data/imgs`, you can
delete the zip file `data/imgs.zip` to save space.
"""

import os
import shutil
from subprocess import call

if os.path.exists('data'):
    shutil.rmtree('data')
os.mkdir('data')

url = 'https://storage.ramp.studio/pollenating_insects_3'
f_names = ['class_codes.csv', 'test.csv', 'train.csv', 'imgs.zip']
for f_name in f_names:
    url_in = '{}/{}'.format(url, f_name)
    f_name_out = os.path.join('data', f_name)
    cmd = 'wget {} --output-document={} --no-check-certificate'.format(
        url_in, f_name_out)
    call(cmd, shell=True)

call('unzip data/imgs.zip', shell=True)
os.rename('data_3/public_imgs', 'data/imgs')
shutil.rmtree('data_3')
