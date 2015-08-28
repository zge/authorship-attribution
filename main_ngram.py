# -*- coding: utf-8 -*-
"""
Main script for generating N-gram models and get perplexity using SRILM
For each course, compute the perplexity per seed, then per order 
(train, test sets are from the same seed partition)

Zhenhao (Roger) Ge, 2015-08-20

"""

# import public modules
import os, sys
import csv
pyver = sys.version_info[0]

# setup directories
directory = {}
path = r'Dropbox/Work/Other/authorship_attribution/codes'
if os.name == 'nt': 
    prefix = 'D:'   
elif os.name == 'posix':
    dirs = ['/home/rge', '/cygdrive/d']
    for url in dirs:
        if os.path.isdir(url):
            prefix = url
            break
directory['work'] =  os.path.join(prefix + os.sep, path).replace('/', os.sep)
directory['root'] =  os.path.dirname(directory['work'])
directory['data'] = os.path.join(directory['root'], 'data')
directory['result'] = os.path.join(directory['root'], 'results')
directory['stem'] = os.path.join(directory['data'], 'stem')
directory['split'] = os.path.join(directory['data'], 'split')
directory['lm'] =  os.path.join(directory['data'], 'lm') 
if not os.path.exists(directory['lm']): os.mkdir(directory['lm'])
directory['current'] = os.getcwd()    
if directory['current'] != directory['work']: os.chdir(directory['work'])
if pyver == 2: print 'current directory:', os.getcwd()
if pyver == 3: print('current directory:', os.getcwd())

# import local modules  
from dataprep import get_files
from srilm import trainlm, perplexity 
  
datasets = get_files(directory['stem'], 'txt') 
datasets = sorted([os.path.split(d)[1] for d in datasets]) 

# specify parameters in looping
num = {}
num['course'] = len(datasets)
num['seed'] = 10
order_range = [1,5]
num['order'] = order_range[1]-order_range[0]

# specify parameter in training and scoring
write = 0; no_sos = 1; no_eos = 1; debug = 0

# specify current dataset and course
for idx0 in range(num['course']):
  dataset = datasets[idx0]
  course = os.path.splitext(dataset)[0]
  
  # train models and find ppls for selected course
  filename = {}
  filename['result'] = os.path.join(directory['result'], course + '_ppl.csv')
  ppl = [[0 for j in range(num['order'])] for i in range(num['seed'])]
  ppl1 = [[0 for j in range(num['order'])] for i in range(num['seed'])]
  for idx1 in range(num['seed']):
  
    # prepare files for SRILM
    filename['train'] = os.path.join('../data/split', course + \
      '_train_rand' + '{0:0>2}'.format(idx1) + '.txt')
    filename['valid'] = filename['train'].replace('_train_', '_valid_')
    #filename['sent'] = os.path.join('../data/lm', course + '_sent.txt')
    
    # read sentences from valid file
    with open(filename['valid'], 'rb') as f: sentences = f.readlines()
    sentences = [s.rstrip() for s in sentences]   
    
    # gen lm and ngram files with input corpus using ngram-count
    for idx2, order in enumerate(range(order_range[0], order_range[1])):
            
        postfix = str(order)+'gram'
        filename['lm'] = (os.path.splitext(filename['train'])[0] + '.lm_' + \
          postfix).replace('split', 'lm')
        if not os.path.isfile(filename['lm']):   
          trainlm(filename['train'], filename['lm'], order, write)
      
        # get perplexity of current LM model on sentences       
        ppl[idx1][idx2], ppl1[idx1][idx2] = perplexity(filename['lm'], \
          sentences, order, no_sos, no_eos, debug)[:2]
        
        # print results
        print 'order', str(order), '->', 'ppl:', str(ppl[idx1][idx2]), \
          'ppl1:', str(ppl1[idx1][idx2])
        print
  
  # write out ppl results for current course into csv file       
  f = open(filename['result'], 'wb')
  wr = csv.writer(f)
  wr.writerows(ppl)
  f.close()
 