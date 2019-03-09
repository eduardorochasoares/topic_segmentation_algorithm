from data_structures import Shot
import numpy as np
from scipy.io import wavfile
import re
from sys import argv
import os, sys
import glob
import evaluate_method
import multiprocessing
import time
import random
import json
sys.path.insert(0, 'document_similarity/')
from document_similarity import DocSim
from gensim.models.keyedvectors import KeyedVectors
from genetic_algorithm import GA
from aubio import source
from aubio import pitch as pt

class Summary:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_file = None
        self.chunks_path = self.video_path + "chunks/"
        self.n_chunks = len(glob.glob(self.chunks_path+ "chunk*"))
        self.chunks = []
        self.video_length = 0

    '''Gets the timestamps of each audio chunk'''
    def extractPauseDuration(self):
        file_path = self.video_path + "seg.txt"
        file = open(file_path, 'r')
        f = file.read()
        times = []
        timesEnd = []
        pause_list = []
        l = re.findall("\+\(\d*\.\d*\)",f )
        for i in l:
            i = i.replace("+","")
            i = i.replace("(","")
            i = i.replace(")","")
            times.append(float(i))

        l = re.findall("\-\(\d*\.\d*\)",f )
        for i in l:
    	    i = i.replace("-","")
    	    i = i.replace("(","")
    	    i = i.replace(")","")
    	    timesEnd.append(float(i))
        file.close()
        pause_list.append(times[0])
        for i in range(1, len(timesEnd)):
            pause_list.append(float(times[i] - timesEnd[i-1]))

        return pause_list, times, timesEnd

    '''Method that create a audio chunk object passing the extracted features'''
    def createShots(self, i, pause, ocr_on, time,end_time,  docSim, prosodic_file):
        pitch = 0
        volume = 0
        try:
            with open(prosodic_file) as f:
                data = json.load(f)
                pitch = float(data[str(i)][0])
                volume = float(data[str(i)][1])

        except FileNotFoundError:
            print('Prosodic features not found')

        s = Shot(i, pitch, volume, pause, [], init_time=time, end_time=end_time)

        s.extractTranscriptAndConcepts(self.video_path, ocr_on, docSim=docSim)

        return s

if __name__ == '__main__':
    stopwords = None
    googlenews_model_path = 'document_similarity/data/GoogleNews-vectors-negative300.bin'
    stopwords_path = "document_similarity/data/stopwords_en.txt"
    docSim = None

    try:
        root_database = sys.argv[1]
    except IndexError:
        print('Please, provide the path from the videolecture to be processed ')
        print('Usage:\n python3 summary path_from_the_video_lecture')
        sys.exit(0)


    '''loads google word embeddings model'''
    with open(stopwords_path, 'r') as fh:
        stopwords = fh.read().split(",")
    model = KeyedVectors.load_word2vec_format(googlenews_model_path, binary=True, limit=1000000)
    docSim = DocSim.DocSim(model, stopwords=stopwords)


    # saves the random seed in the seeds.txt file
    seed  =  random.randrange(sys.maxsize)
    seed_file = open("seeds.txt", 'a')
    seed_file.write(str(seed) + '\n')
    seed_file.close()
    random.seed(seed)


    start_time = time.time()

    print(root_database)
    summary = Summary(root_database)
    ocr_on = False
    pauses, times, times_end = summary.extractPauseDuration()
    duration = [times_end[i] - times[i] for i in range(len(times))]
    summary.video_length = np.sum(duration)
    chunks = []
    summary.n_chunks = len(times)
    '''create the audio chunks structure'''
    for i in range(summary.n_chunks):
        chunks.append(summary.createShots(i, pauses[i], ocr_on, times[i], times_end[i], docSim, summary.video_path + "prosodic.json"))

    old_chunks = chunks
    chunks = [s for s in chunks if s.valid_vector]

    summary.chunks = chunks
    summary.n_chunks = len(chunks)

    boundaries = []
    if summary.n_chunks < 2:
        boundaries = [0]
    else:
        '''calls the genetic algorithm'''
        ga = GA.GeneticAlgorithm(population_size=100, constructiveHeuristic_percent=0.3, mutation_rate=0.05,
                                 cross_over_rate=0.4, docSim=docSim, shots=summary.chunks,
                                 n_chunks=summary.n_chunks, generations=500, local_search_percent=0.3,
                                 video_length=summary.video_length, stopwords=stopwords, ocr_on=ocr_on)
        boundaries = ga.run()

    '''print the indexes of the points that are topic boundaries'''
    print(boundaries)
