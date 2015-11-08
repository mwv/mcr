=================================
Monkey Call Recognition
=================================

Source code for training, decoding and evaluating an automatic monkey call recognizer.

* Free software: GPL3 license

Features
--------

* Classification and decoding of calls in isolation
* Automatic transcription of audio files

Installation
------------

Install python 2.7.x using your system's package manager.

Optionally, install virtualenv to keep separate the required packages and then do::

  $ virtualenv -p $(which python2.7) ~/mcr_venv && source ~/mcr_venv/bin/activate

Next, install the required packages::

  $ pip install -r requirements.txt

And you're good to go.

=====
Usage
=====

The system consists of two sets of scripts with similar interfaces, for `segmented` recognition and `transcription` respectively. Both sets can be found in the `src` folder. We will discuss these scripts below. First, we need to discuss the annotation and configuration formats and

Annotation files
----------------

The `train`, `predict` and `eval` scripts all require an annotation file. The annotation file is a csv-file with `filename`, `start` and `end` columns and an optional `label` column which is needed only for the `train` and `eval` scripts. The rows in the annotation file point to segments of audio files, denoted by a filename, start and end times and possible call label. As an example, suppose we have two one-second audio recordings, `/home/me/A.wav` and `/home/me/B.wav` and we have annotated a `PYOW` call in file `A` at the time interval `0.100` - `0.200` and a `HACK` call in file `B` at the time interval `0.500` - `0.600`; we then make an annotation file (`annotation.csv`) with the following contents to train our classifier:::

  filename,start,end,label
  /home/me/A.wav,0.000,0.100,SIL
  /home/me/A.wav,0.100,0.200,PYOW
  /home/me/A.wav,0.200,1.000,SIL
  /home/me/B.wav,0.000,0.500,SIL
  /home/me/B.wav,0.500,0.600,HACK
  /home/me/B.wav,0.600,1.000,SIL

where we used `SIL` to denote the absence of calls. Note that, although we have two audio files, we only make a single annotation file for the classifier script. In case we have a trained classifier available and wish to use it to transcribe a new one-second recording `C.wav` we have made and haven't manually annotated, we can make an annotation file without a `label` column:::

  filename,start,end
  /home/me/C.wav,0.000,1.000

Configuration files
-------------------

The `train` scripts require a configuration file which lists the settings for the feature extractors and the hyperparameters for the classifiers. This file needs to be in `toml` format. Examples of these files with default settings can be found in `src/segmented.cfg` and `src/transcription.cfg`. These example files list all the parameters to the system, so if you wish to play around with different settings, simply changing these configuration files is a good option.

Segmented classification
------------------------

*Segmented classification* refers to classifying individual monkey calls in isolation, i.e. we know their exact location in an audio stream and need to learn their identity. The following scripts are supplied:

- `src/segmented_train.py`
- `src/segmented_predict.py`
- `src/segmented_eval.py`

For example, to train a classifier using the above annotated calls, we do:::

  $ python segmented_train.py annotation.csv segmented.cfg trained.clf

Note that this will only consider the annotated *calls*, the script will ignore any non-call (`SIL`) segments in the audio files.

To use this trained classifier to predict the annotations again, we do:::

  $ python segmented_predict.py annotation.csv trained.clf predicted.csv

To compare the predicted labels to the annotation, we do:::

  $ python segmented_eval.py annotation.csv predicted.csv

Which will print scores to the standard output.

For more information about how to use the scripts, see their help messages.

Transcription
-------------

*Transcription* refers to the automatically transcribing entire audio files, marking where calls occur and their identity and where there is only silence (or, more likely, background noise). The following scripts implement a complete system for training, decoding and evaluating a transcription classifier:

- `src/transcription_train.py`
- `src/transcription_predict.py`
- `src/transcription_eval.py`

The scripts for complete transcription of audio files follow the same conventions as those above. They will however take considerably longer to run.
