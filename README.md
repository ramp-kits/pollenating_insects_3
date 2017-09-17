# Bee-o-diversity challenge  RAMP

Authors: Mehdi Cherti & Balazs Kegl

[![Build Status](https://travis-ci.org/ramp-kits/pollenating_insects_3.svg?branch=master)](https://travis-ci.org/ramp-kits/pollenating_insects_3)

## Getting Started

An Ubuntu 16.04 AMI image named `pollenating_insects_users_3` has been
made available at the Oregon site of
[AWS](https://us-west-2.console.aws.amazon.com).
We used it with `g3.4xlarge` instances, but it may work with other GPU
insances as well.

After launching the instance and logging in, simply run

```
cd pollenating_insects_3
ramp_test_submission
```

or

```
ramp_test_submission --submission <submission>
```

A [dedicated notebook](pollenating_insects_3_starting_kit.ipynb)
is avaiable to get you started on the problem.

**The offical competition rules are also in this notebook. You accept these
rules automatically when you make a submission at the
[RAMP site](http://www.ramp.studio/events/pollenating_insects_3_JNI_2017).**

### Ramp overview

The library [`ramp-workflow`](https://github.com/paris-saclay-cds/ramp-workflow)
contains tools to define data challenges and a script to test submissions.
As a participant all you only need to know is that the RAMP workflow loads
and test the files in `submissions/<submission>/`.

For this competition you need to submit two files:

1. `batch_classifier.py` containing your model. It should contain a class
implementing `fit` and `predict_proba` methods.
2. `image_preprocessor.py`. It should contain a function named `transform`
and optionally `transform_test`.


Go to [`ramp-workflow`](https://github.com/paris-saclay-cds/ramp-workflow)
for more help on the [RAMP](http:www.ramp.studio) ecosystem.

### Making a submission

To make a submission you first need to [sign up](https://www.ramp.studio/sign_up)
to the RAMP site, then sign up to the
[challenge event](http://www.ramp.studio/events/pollenating_insects_3_JNI_2017). 
Both sign-ups need approval, so be patient. Once you are approved, you can 
submit `batch_classifier.py` and `image_preprocessor.py` in your
[sandbox](http://www.ramp.studio/events/pollenating_insects_3_JNI_2017/sandbox).

**Before making a submission, please check that your code will properly
run on the backend by running**:

```
ramp_test_submission
```

or

```
ramp_test_submission --submission <submission>
```

## Experimeting on your own setup

You can also run experiments on your own setups. To do so please do as
following.

**Bear in mind that your submission we be run on our backend and
using non supported libraries will make submission fail.**

$ git clone https://github.com/ramp-kits/pollenating_insects_3

### Downloading data

Download the data (~17GB) by running

```
python download_data.py
```

the first time. It will create `data/imgs` and download the images there
using the names `<id>`, where `<id>`s are coming from `data/train.csv`
and `data/test.csv`. If images are properly unzipped in `data/imgs`,
you can delete the zip file `data/imgs.zip` to save space.

### Installing dependencies

The installation script [`install.sh`](install.sh) used to make the AMI
is also available. Depending on your current installation, you may not need
to execute all of this, but it shows the versions of the various libraries
against which we tested the starting kit.

### Keras channel

You should set `image_data_format` to `channels_last` in `~/.keras/keras.json`.
