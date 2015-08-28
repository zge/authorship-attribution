# -*- coding: utf-8 -*-
"""
Main script to get classification of Authorship Attribution (AA) project
using N-gram models

Zhenhao (Roger) Ge, 2015-08-22
"""

# import public modules
import os, sys, time, pickle
import numpy as np
#import matplotlib.pyplot as plt
#from itertools import cycle
pyver = sys.version_info[0]

# find path prefix based on op system
path = r'Dropbox/Work/Other/authorship_attribution/codes'
if os.name == 'nt': 
    prefix = 'D:'   
elif os.name == 'posix':
    dirs = ['/home/rge', '/cygdrive/d']
    for url in dirs:
        if os.path.isdir(url):
            prefix = url
            break
          
# setup directories
directory = {}          
directory['work'] =  os.path.join(prefix + os.sep, path).replace('/', os.sep)
directory['root'] =  os.path.dirname(directory['work']) 
directory['data'] = os.path.join(directory['root'], 'data')
directory['result'] = os.path.join(directory['root'], 'results')
directory['stem'] = os.path.join(directory['data'], 'stem')
directory['split'] = os.path.join(directory['data'], 'split')
directory['lm'] =  os.path.join(directory['data'], 'lm')
directory['current'] = os.getcwd()    
if directory['current'] != directory['work']: os.chdir(directory['work'])
if pyver == 2: print 'current directory:', os.getcwd()
if pyver == 3: print('current directory:', os.getcwd())

# import local modules  
from dataprep import get_files
from srilm import perplexity 
from classify import nbest_accuracy, confusion_array, aggregate
  
datasets = get_files(directory['stem'], 'txt') 
datasets = sorted([os.path.split(d)[1] for d in datasets])
courses = [os.path.splitext(dataset)[0] for dataset in datasets] 

# specify parameters in looping
num = {}
num['course'] = len(datasets)
num['seed'] = 10
order_range = [1,5]
num['order'] = order_range[1]-order_range[0]

# specify parameter in scoring
no_sos = 1; no_eos = 1
show_progress_after = 100
group_len_max = 50

# start looping over course and seed
for idx0 in range(num['course']):
  
  course = courses[idx0]
  print 'processing course', str(idx0)+':', course, '...' 
  
  for idx1 in range(num['seed']):     
    
    filename = {}
    filename['test'] = os.path.join('../data/split', course + \
          '_test_rand' + '{0:0>2}'.format(idx1) + '.txt')
    testname = os.path.split(filename['test'])[1]
    filename['score'] = os.path.join(directory['result'], 'scores', \
      os.path.splitext(testname)[0] + '.pickle')
    filename['nbest'] = os.path.join(directory['result'], 'nbest', \
      os.path.splitext(testname)[0] + '.csv')
    filename['confusion'] = os.path.join(directory['result'], 'confusion', \
      os.path.splitext(testname)[0] + '.csv')  
    print 'processing seed', str(idx1)+':', filename['test'], '...' 
      
    # get sentences and its number  
    sentences = file(filename['test']).readlines()
    sentences = [s.rstrip() for s in sentences]
    num['sent'] = len(sentences)
    
    # load or compute ppl per order, sentence, course
    if os.path.exists(filename['score']):
      
      # load ppl
      with open(filename['score']) as f: 
        ppl = pickle.load(f)[0]
      
    else:
    
      # compute ppl
      ppl = np.zeros([num['order'], num['sent'], num['course']])
      timecost = np.zeros([num['order'], num['sent']])
      start_time_chunk = time.time()
      for i, order in enumerate(range(order_range[0], order_range[1])):
        for j, sent in enumerate(sentences):
          start_time = time.time()    
          for k, cour in enumerate(courses):
            filename['lm'] = os.path.join(directory['lm'], cour + '_train_rand' + \
              '{0:0>2}'.format(idx1) + '.lm_' + str(order) + 'gram')
            ppl[i][j][k] = perplexity(filename['lm'], sent, order, no_sos, no_eos)[0]
          timecost[i][j] = time.time() - start_time
          if np.mod(j+1, show_progress_after) == 0:
            print 'order:', str(order) + ',', 'sent. idx: ', j-show_progress_after+1, \
              'to', j, 'processed', '(', '%.2f' % (time.time() - start_time_chunk) , 'sec ) ...'
            start_time_chunk = time.time()  
      
      # Saving variable(s)
      if not os.path.isdir(os.path.dirname(filename['score'])):
        os.mkdir(os.path.dirname(filename['score']))  
      with open(filename['score'], 'w') as f:
        print 'saving', filename['score'], '...'
        pickle.dump([ppl, timecost], f)
    
    # compute nbest accuracy (# of course (N) x order x group length max)
    exist = {}
    exist['nbest'] = os.path.exists(filename['nbest'])
    exist['confusion'] = os.path.exists(filename['confusion'])
    if not ( exist['nbest'] and exist['confusion']) :
      #acc = np.zeros([num['order'], group_len_max]) # acc is nbest[:,:,0]
      nbest = np.zeros([num['course'], num['order'], group_len_max])
      confusion = np.zeros([num['course'], num['order'], group_len_max])
      for i in range(num['order']):    
        for j, gl in enumerate(range(1, group_len_max+1)):
          ppl_agg = aggregate(ppl[i], gl)
          #acc[i,j] = get_accuracy(ppl_agg, idx0)
          if not exist['nbest']: 
              nbest[:,i,j] = nbest_accuracy(ppl_agg, idx0)
          if not exist['confusion']:
              confusion[:,i,j] = confusion_array(ppl_agg)
      
      if not exist['nbest']:
        # flat nbest accuracy to 2D
        nbest_2d = np.zeros([num['course']*num['order'], group_len_max+2])
        cnt = 0
        for i in range(num['course']):
          for j in range(num['order']):
            nbest_2d[cnt, 0:2] = [i+1, j+1]
            nbest_2d[cnt, 2:] = nbest[i,j,:]
            cnt += 1
      
        # save nbest accuracy
        if not os.path.isdir(os.path.dirname(filename['nbest'])):
          os.mkdir(os.path.dirname(filename['nbest']))
        print 'saving', filename['nbest'], '...'  
        np.savetxt(filename['nbest'], nbest_2d, delimiter=",")
      
      if not exist['confusion']:      
        # flat confusion 3D matrix to 2D
        confusion_2d = np.zeros([group_len_max*num['order'], num['course']+2])
        cnt = 0
        for i in range(group_len_max):
            for j in range(num['order']):
                confusion_2d[cnt, 0:2] = [i+1, j+1]
                confusion_2d[cnt, 2:] = confusion[:,j,i]
                cnt += 1
        
        # save confusion
        if not os.path.isdir(os.path.dirname(filename['confusion'])):
            os.mkdir(os.path.dirname(filename['confusion']))
        print 'saving', filename['confusion'], '...'
        np.savetxt(filename['confusion'], confusion_2d, delimiter=",")
    
      ## show nbest accuracy (n = 1,2,3)
      #nbest_min = np.min(nbest)
      #y_min = np.floor(nbest_min*10)/10
      #lines = ["-","--","-.",":"]
      #linecycler = cycle(lines)
      #for i in range(3):
      #  plt.figure(i+1)
      #  for j in range(num['order']): 
      #    plt.plot(np.arange(group_len_max)+1, nbest[i,j,:], next(linecycler))
      #  plt.legend(['unigram', 'bigram', 'trigram', '4gram'], loc='lower right')
      #  plt.xlabel('number of sentences (k)')
      #  plt.ylabel('accuracy (k sentences)')
      #  plt.title('N-best accuracy, N =' + str(i+1))
      #  plt.ylim((y_min,1))
      #  plt.show()
      #
      ## close all figures  
      #plt.close("all")  

