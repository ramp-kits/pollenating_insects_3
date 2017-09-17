# RAMP starting kit on the MNIST dataset

Authors: Mehdi Cherti & Balazs Kegl

[![Build Status](https://travis-ci.org/ramp-kits/pollenating_insects_2.svg?branch=master)](https://travis-ci.org/ramp-kits/pollenating_insects_2)

Go to [`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow) for more help on the [RAMP](http:www.ramp.studio) ecosystem.

After cloning, run

```
python download_data.py
```

the first time. It will create `data/imgs` and download the images there
using the names `<id>`, where `<id>`s are coming from `data/train.csv` and `data/test.csv`. If images are properly unzipped in `data/imgs`, you can delete the zip file `data/imgs.zip` to save space.

We have included an [`install.sh`](install.sh) script that we used to set up a fresh Ubuntu 16.04 server with GPU. Depending on your current installation, you may not need to execute all of this, but it shows the versions of the various libraries against which we tested the starting kit.

You should set `image_data_format` to `channels_last` in `~/.keras/keras.json`.

Install ramp-workflow (rampwf), then execute

```
ramp_test_submission
```

or

```
ramp_test_submission --submission <submission>
```

to execute other example submissions from the folder `submissions`.

We have also preinstalled the starting kit and the data on the Oregon site of [AWS](https://us-west-2.console.aws.amazon.com), called `pollenating_insects_users_3`. We used it with `g3.4xlarge` instances, but it may work with other GPU insances as well. After launching the instance, simply

```
cd pollenating_insects_3
ramp_test_submission --submission <submission>
```

Get started on this RAMP with the [dedicated notebook](pollenating_insects_3_starting_kit.ipynb). The notebook also contains the competition rules.
