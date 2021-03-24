## Script by ONewman and EAvcu

#This script reads an excel file that includes all the words and extract two second clips from SWC/english/folders,
# mix the two seconds clip with three different noise based on random but controlled SNR values, and write each word
# to a folder (under which there will be clean wav files and a noise folder that has all the wav files mixed with noise) an Output folder.


import sys
import lxml.etree as ET
import os
import random
import xlrd
import math
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# wordList = ["popular"] # to extract all occurences of only one word
fname = os.path.join("Path_to_the_folder/english/*/aligned.swc") #path to the main swc folder
# fname = "/english/Age_of_Empires/aligned.swc" # to extract a word from only one topic

# word matrix variables 
limit = 450 #top limit of each occurence of all the words

wordListFile = '/Path_to_the_folder/wordlist.xlsx'
outputDir = '/Path_to_the_folder/Output/'


wordListWorkBook = xlrd.open_workbook(wordListFile)
wordListSheet = wordListWorkBook.sheet_by_name('Sheet1')
wordList = wordListSheet.col_values(0)

noiseDir = '/Path_to_the_folder/Noise'
noiseFolders = os.listdir(noiseDir)
noiseFolders = noiseFolders[1:]


def extract_from_file(word, fname):
    root = ET.parse(fname)
    start_times = root.xpath('//n[@pronunciation="' + word + '"]/@start')
    end_times = root.xpath('//n[@pronunciation="' + word + '"]/@end')
    return start_times, end_times


# given a signal, noise (audio) and desired SNR, this gives the noise (scaled version of noise input) that gives the desired SNR
def get_noise_from_sound(signal, noise, SNR):
    RMS_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, SNR / 10)))

    # current RMS of noise
    RMS_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (RMS_n / RMS_n_current)

    return noise


if __name__ == "__main__":
    for word in wordList:
        wordDir = outputDir + '/' + word
        wordNoiseDir = wordDir + '/noise'
        wordCount = 1;
        if not os.path.exists(wordDir):
            os.mkdir(wordDir)
            print("Directory ", wordDir, " Created")

        else:
            print("Directory ", wordDir, " Already Created")

        if not os.path.exists(wordNoiseDir):
            os.mkdir(wordNoiseDir)
            print("Directory ", wordNoiseDir, " Created")
            notExists = True
        else:
            print("Directory ", wordNoiseDir, " Already Created")
            notExists = False

        if notExists:
            for fname in sys.argv[1:]:
                print(fname)
                oggFile = fname[
                          :-12] + '/mono_audio.ogg'  # -we are removing /aligned.swc from the file name so we can then reference the ogg file			
                start, end = extract_from_file(word, fname)
                print(start, end)
                for index in range(0, len(start)):
                    if wordCount > limit:
                        break;
                    else:
                        print('THE WORD EXAMPLE COUNT IS CURRENTLY: ' + str(wordCount))
                        print(index)
                        print(oggFile)
                        startTime = start[index]
                        endTime = end[index]
                        print(startTime, endTime)
                        oggWordFile = oggFile
                        wavFileNameWithoutWav = wordDir + fname[49:-12] + word + str(index)
                        wavFileNameWithoutWav = str(wavFileNameWithoutWav)
                        oggWordFile = str(oggWordFile)
                        # print(startTime)
                        startTime = int(startTime)  # time in seconds 
                        endTime = int(endTime)  # time in seconds 
                        print(startTime, endTime)
                        lengthTime = int(endTime - startTime)  # time in seconds 
                        binNum = int(lengthTime / 10)
                        jitterStart = int(0 - (lengthTime / 2))
                        jitterEnd = jitterStart + binNum
                        averageTime = (startTime + endTime) / 2000  # time in ms
                        print("NEXT IS BINS")
                        # wordCount+=1

                        for i in range(1, 11):
                            # Applies the 10 different bins of random jitter
                            print("BIN " + str(i))
                            randomJitterNum = random.randrange(jitterStart, jitterEnd,
                                                               1)  # randomizes the jitter ms with 1 ms step increments, all in second format
                            randomJitterNum = randomJitterNum / 1000  # in seconds 
                            calculation = averageTime + randomJitterNum - 1  # add the jitter to the original start time 
                            calculation = str(calculation)  # changes start time to a string for command later.
                            wavFileName = wavFileNameWithoutWav + '-' + str(i) + '.wav'
                            wavFileName = str(wavFileName)
                            stringCommand = 'sox' + ' ' + oggWordFile + ' ' + wavFileName + ' trim ' + calculation + ' 2'
                            os.system(stringCommand)  # creates the wav file
                            # updates the bin numbers for the jitter
                            jitterStart = jitterStart + binNum
                            jitterEnd = jitterEnd + binNum

                            # Creates the Noise Files
                            randomFolderNum = random.randrange(0, len(noiseFolders) - 1, 1)
                            randomFolder = noiseFolders[randomFolderNum]
                            randomFolder = noiseDir + '/' + str(randomFolder)
                            folderFiles = os.listdir(randomFolder)
                            randomFileNum = random.randrange(0, len(folderFiles) - 1, 1)
                            randomNoiseFile = folderFiles[randomFileNum]
                            randomNoiseFile = randomFolder + '/' + str(randomNoiseFile)

                            # Load signal and noise
                            signal_file = wavFileName  # loads in the jitter wav file we just created
                            signal, sr = librosa.load(signal_file)
                            signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))

                            noise_file = randomNoiseFile
                            print(noise_file)
                            noise, sr = librosa.load(noise_file)
                            noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))

                            # Gaussian random number generation
                            # mu (mean) = -3 , sigma = 2 (SD) # mean and standard deviation for auditory scenes and music
                            # mu (mean) = +3 , sigma = 2 (SD) # mean and standard deviation for speech babble
                            if randomFolderNum == 0:
                                mu, sigma = 10, 2  # mean and standard deviation for speech babble
                            else:
                                mu, sigma = 7, 2  # mean and standard deviation for auditory scenes and music

                            SNR_G = np.random.normal(mu, sigma, None)
                            print(SNR_G)

                            noise = get_noise_from_sound(signal, noise, SNR=SNR_G)
                            signal_noise = signal + noise

                            # save
                            fileName = wordNoiseDir + '/' + word + str(wordCount) + 'noise' + 'bin' + str(i) + '.wav'
                            write(fileName, sr, signal_noise)
                    wordCount += 1
                print(wordDir)
        else:
            continue 

            # oggFile = "C:\\Users\\mimi\\PycharmProjects\\GowLab\\english\\Age_of_Empires\\audio.ogg"

# data, samplerate = sf.read('C:/Users/mimi/PycharmProjects/GowLab/english/Age_of_Empires/audio.ogg')
# newWav = sf.write('new_file.wav', data, samplerate)

# file = AudioSegment.from_wav(newWav)
# file = AudioSegment.from_ogg(oggFile)
# slice middle eight seconds of audio


# left_four_seconds = (midpoint - 4) * 1000 #pydub workds in milliseconds
# right_four_seconds = (midpoint + 4) * 1000 #pydub workds in milliseconds
# startTime = 112570
# endTime = 112582

# wordSlice = file[startTime:endTime]

# Play slice
# play(wordSlice)

# or save to file
# wordSlice.export("newFile.oog", format="oog")



