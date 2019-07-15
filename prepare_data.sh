#!/usr/bin/env bash
#
# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# echo "Downloading SVHN"
# mkdir -p data/images/svhn
# (
#     cd data/images/svhn
#     curl -O 'http://ufldl.stanford.edu/housenumbers/{train,test,extra}_32x32.mat'
# )

# echo
# echo "Downloading CIFAR-10"
# mkdir -p data/images/cifar/cifar10
# (
#     cd data/images/cifar/cifar10
#     curl -O 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz'
#     tar xvzf cifar-10-matlab.tar.gz
#     mv cifar-10-batches-mat/* .
#     rmdir cifar-10-batches-mat
# )

# echo
# echo "Preprocessing CIFAR-10"
# python prepare_data/preprocess_cifar10.py

# echo
# echo "All done!"


echo "Preparing google audio datasets"
mkdir -p data/images/audio/download/speech_commands
(
    cd data/images/audio/download
    wget -O speech_commands_v0.01.tar.gz "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    tar -xvzf speech_commands_v0.01.tar.gz -C ./speech_commands_v0.01/ 
    )
echo "google command datasets downloaded and unziped"

echo "convert wav format to png format"
python prepare_data/audio2img_google.py

echo "Preprocessing audio"
python prepare_data/preprocess_audio30.py

cd ../../../../

echo "Preparing urban datasets"
mkdir -p data/images/audio/download
(
    mv urban-sound-classification data/images/audio/download
    cd data/images/audio/download
    unzip urban-sound-classification.zip -d urban-sound-classification
    cd urban-sound-classification 
    unzip train.zip
    # unzip test.zip
    )
echo "urban dataset downloaded and unziped"

echo "convert wav format to png format"
python prepare_data/audio2img_urban.py

echo "Preprocessing audio"
python prepare_data/preprocess_urban.py

echo "All done!"
