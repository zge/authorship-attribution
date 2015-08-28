# -*- coding: utf-8 -*-
"""
Collection of functions related to authorship attribution classification

Zhenhao (Roger) Ge, 2015-08-22
"""

import numpy as np

def get_accuracy(ppl, target):
  """
  get the avg. accuracy from (sentence-level) perplexities
  inputs: 
    ppl: perplexity [# sentences x # hyp courses]
    target: the index of the target course
  """
  
  ppl_min = np.min(ppl, 1)
  flag = ppl_min == ppl[:, target]
  accuracy = float(np.sum(flag)) / len(flag)
  
  return accuracy


def nbest_accuracy(ppl, target):
  """
  find the avg. nbest accuracy among all sentences 
  """  
  
  num_sent, num_course = np.shape(ppl)
  pos = np.zeros(num_sent)
  for i in range(num_sent):
    s = ppl[i,:]
    idx_sorted = sorted(range(len(s)), key=lambda k: s[k])
    pos[i] = idx_sorted.index(target)
  
  nbest = np.zeros(num_course)  
  for i in range(num_course):
    nbest[i] = float(np.sum(pos <= i)) / num_sent
    
  return nbest
  
def confusion_array(ppl):
  """
  get the min of each row and indicate in confusion matrix
  1 for the index of min, o.w. 0
  then sum up w.r.t rows and normalize to get the prob. 
  return confusion array as a row vector      
  """
  
  num_sent, num_course = np.shape(ppl)
  conf_mtx = np.zeros([num_sent, num_course])
  for i in range(num_sent):
    idx = min((val, idx) for (idx, val) in enumerate(ppl[i,:]))[1]
    conf_mtx[i, idx] = 1
  conf_array = np.sum(conf_mtx, 0) / num_sent   
  return conf_array     
  
  
def aggregate(ppl, group_len):
  """
  group sentence-level ppl with specified group length
  inputs:
    ppl: perplexity [# sentences x # hyp courses]
    group_len: # of sentences to be grouped
  """ 
  
  # get the size parameters of ppl
  num_sent, num_course = np.shape(ppl)
  
  # compare sentence group length with the number of sentences
  if group_len > num_sent:
    print 'sentence group length cannot be greater than the # of sentences', \
      num_sent, '!'
    return 
  
  ## method 1 is supposed to be faster but a bit slower than method 2 
  #ppl_agg = np.zeros([num_sent-group_len+1, num_course])
  #ppl_agg[0,:] = np.sum(ppl[:group_len, :], 0) 
  #for i in range(1, num_sent-group_len+1):
  #  ppl_agg[i,:] = ppl_agg[i-1,:] - ppl[i-1,:] + ppl[i+group_len-1,:]
 
  # method 2 is a little faster and easier to understand 
  ppl_agg = np.zeros([num_sent-group_len+1, num_course])
  for i in range(num_sent-group_len+1):
    ppl_agg[i,:] = np.sum(ppl[i:i+group_len,:], 0)
    
  return ppl_agg


## tests
#acc1 = get_accuracy(aggregate(ppl, 1), target)
#acc2 = get_accuracy(aggregate(ppl, 3), target)  

my_list = [3,2,5,7]
tmp = [(val, idx) for (idx, val) in enumerate(my_list)]
val, idx = min((val, idx) for (idx, val) in enumerate(my_list))