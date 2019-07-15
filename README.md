# Semi-supervised Audio Classification with Consistency-Based Regularization

This is the TensorFlow source code for the "Semi-supervised Audio Classification with Consistency-Based Regularization" paper.

The environment can be found in dockerhub:

`docker pull loklu/mt_tensorflow:tf1.2.1_py35_lib2`

To prepare the dataset, please download the [urban sound data](https://www.kaggle.com/pavansanagapati/urban-sound-classification) and save it under the main folder("./SS_Audio_Classification/"). Then run the following script:

`sh prepare_data.sh`

To train the model, run:

`cd audio_mt_public`

`sh experiments/script.sh`

The code is built on the [official code](https://github.com/CuriousAI/mean-teacher) of paper "Mean teachers are better role models".
