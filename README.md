# cnn_duallexicon
This Repository contains the sample code or scripts used in the CNN Dual Lexicon project developed by the [Gow Lab](https://gowlab.mgh.harvard.edu/), Department of Neurology, Massachusetts General Hospital/Harvard Medical School.

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
We extracted two second clips for each occurrence of a target word and these two second clips were mixed with three different background noises (three different noise) with randomly assigned SNR levels (see the preprint paper for the details of these process). The noise files were generated by using the following repositories: (i) auditory scenes, (ii) instrumental music, (iii) multi-speaker speech babble. We used the 2013 IEEE AASP [Challenge on Detection and Classification of Acoustic Scenes and Events corpus](http://c4dm.eecs.qmul.ac.uk/sceneseventschallenge/description.html)  (Stowell et al., 2015) for auditory scenes and corpus of public domain audiobook recordings (https://librivox.org/) to create the multi speaker speech babble. For the instrumental music, we used the Instrument Recognition in Musical Audio Signals (IRMAS) corpus (Bosch et al., 2012) which  includes predominant instruments like cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin.

### Command
```
python extract_words.py
```

## Step 2: Cochleagram Creation
We used cochleagrams of each two second clip as the input to the network. A cochleagram is very similar to a spectrogram which is used to represent audio signals in the time frequency domain. Cochleagrams were created using the code shared by [Feather et. al. (2019)](https://github.com/jenellefeather/tfcochleagram). Each two second clip was passed through a bank of 203 bandpass filters resulting a cochleagram representation of 203 x 400 (frequency x time). See figure below for a schematic representation of training data preparation.

### Command
```
python tfcochleagram_ns.py
```
![image](https://user-images.githubusercontent.com/32641692/112358864-17b38480-8ca7-11eb-8489-323c2792469a.png)

## Step 3: CNNs
We used CNNs comprised of convolution, normalization, pooling and fully connected layers (see Kell et. al., 2018 for the definitions of the operations of each layer). In addition, we used Gaussian Noise layer which apply zero centered Gaussian noise (0.1) after each convolutional layer during training. Our CNN architecture consisted of 21 layers (see below) with the final softmax classification layer. After the fully connected layer, a dropout (0.1) layer was also used. Finally, we leveraged early stopping to avoid overfitting (when the validation loss does not decrease after ten consecutive epochs, the training stops and classifier weights from the epoch that has the lowest validation loss were saved).
Both the dorsal and ventral networks were identical in terms of architectural parameters except for the softmax layer (for dorsal task n=178 and for ventral task n=10). 

### Network Architecture
-	Input (203x400): Cochleagram: 203 frequency bins x 400 time bins
-	Conv1 (68x134x96): Convolution of 96 kernels with a kernel size of 9 and a stride of 3
-	Gaus1 (68x134x96): Gaussian noise (0.1)
-	Norm1 (68x134x96): Normalization over 5 adjacent kernels
-	Pool1 (34x67x96): Max pooling over window size of 3x3 and a stride of 2
-	Conv2 (17x34x256): Convolution of 256 kernels with a kernel size of 5 and a stride of 2
-	Gaus2 (17x34x256): Gaussian noise (0.1)
-	Norm2 (17x34x256): Normalization over 5 adjacent kernels
-	Pool2 (9x17x256): Max pooling over a window size of 3x3 and a stride of 2
-	Conv3 (9x17x512): Convolution of 512 kernels with a kernel size of 3 and a stride of 1
-	Gaus3 (9x17x512): Gaussian noise (0.1)
-	Norm3 (9x17x512): Normalization over 5 adjacent kernels
-	Pool3 (5x9x512): Max pooling over a window size of 3x3 and a stride of 2
-	Conv4 (5x9x1024): Convolution of 1024 kernels with a kernel size of 3 and a stride of 1
-	Gaus4 (5x9x1024): Gaussian noise (0.1)
-	Norm4 (5x9x1024): Normalization over 5 adjacent kernels
-	Conv5 (5x9x512): Convolution of 512 kernels with a kernel size of 3 and a stride of 1
-	Gaus5 (5x9x512): Gaussian noise (0.1)
-	Norm5 (5x9x512): Normalization over 5 adjacent kernels
-	Pool4 (3x5x512): Mean pooling over a window size of 3 and a stride of 2
-	Dense1 (4096): A fully connected layer
-	Dense2 (178 or 10): A fully connected layer before the softmax function for words (n = 178) or semantic domains (n = 10).


### Command
```
python network.py
```
![image](https://user-images.githubusercontent.com/32641692/112854090-7bf59000-907b-11eb-8ed6-8d0c04c6cecc.png)

## Step 4: Generalization Tests
For the articulatory generalization task, we used two tasks: (i) onset phoneme category (ii) syllable length. Onset phoneme category task was used to determine whether the first phoneme in a word was a fricative, nasal, stop, liquid, or vowel/glide (5 categories with 1050 exemplars each). Syllable length test was used to test whether words consisted  of one, two, three or four syllables (4 categories 900 exemplars each). For the semantic tasks, we again use two tasks: (i) the animacy and (ii) concreteness. The animacy task classified  nouns as animate or inanimate (2 categories with 400 exemplars each). The concreteness task classified nouns as being abstract or concrete (2 categories with 500 exemplars each).

### Command
```
python generalization.py
```

![image](https://user-images.githubusercontent.com/32641692/112855967-579ab300-907d-11eb-8b78-45e6859616cb.png)
