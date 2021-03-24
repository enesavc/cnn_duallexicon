# cnn_duallexicon
This Repository contains the sample code or scripts used in the CNN Dual Lexicon project

## CNN Dual Lexicon Preprint

Avcu, E., Newman, O., Gow, D. (2021). A Tale of Two Lexica: Testing Computational Hypotheses with Deep Convolutional Neural Networks. arXiv preprint arXiv:XXXXXX.XXXXXX.

## Link to Talks

### Link to our Society of Neurobiology of Language Poster Presentation
https://www.youtube.com/watch?v=nfQbnTl9hbY

### Link to our Psychonomics Society Talk
https://www.youtube.com/watch?v=mfVCsV8wSOg

## Environment Specs
The following specs are for a CentOS Linux system.
It is suggested to set up a conda environment (optional), activate your environment, install python, tensorflow, or anyother packages into your specific environment.

Example Code
```
conda create --prefix ./ns/gpu
conda activate /PATH/ns/gpu
conda install --prefix ./gpu python=3.8.5
conda install --prefix ./gpu tensorflow-gpu==2.4.1
```

## Step 1: Preparing the Training Data
We have used the [Spoken Wikipedia Corpus](https://nats.gitlab.io/swc/) (SWC) to extract the audio files. We used words that has at least four characters long and occurred between 200 and 450 times. We have used 178 words for this preliminary task.
We extracted two second clips for each occurrence of a target word and these two second clips were mixed with three different background noises with randomly assigned SNR levels (see the preprint paper for the details of these process).
