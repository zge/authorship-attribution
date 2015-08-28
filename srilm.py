# -*- coding: utf-8 -*-
"""
Collection SRILM related functions
"""

import os, re, subprocess

def trainlm(trainfile, modelfile, order=3, write=1, addsmooth=10):
  """
  train LM using SRILM
  parameters:
    - order (3): model order
    - addsmooth (10): adding delta (0-9) to smooth model (this is a poor
      smooth, only for instructional purposes, set 10 to disable it)
    - write (0): flag to write (1) or not write (0) gram file  
  """
  
  # make up command
  command = ['ngram-count', '-text', trainfile, '-order', str(order), \
    '-lm', modelfile]  
  if addsmooth < 10 and addsmooth >= 0:
    command += ['-addsmooth', str(addsmooth)]
  if write == 1:
    gramfile = os.path.splitext(modelfile)[0] + '.%dgram' % order
    command += ['-write', gramfile]

  # run ngram-count in SRILM
  subprocess.check_output(command)
  
  ## example
  #subprocess.check_output(['ngram-count', '-text', filename['train'], \
  #  '-order', str(order), '-addsmooth', str(addsmooth), '-lm', \
  #  filename['lm'], '-write', filename['gram']])
  
  # print out generated files
  print 'file(s) generated:'
  print '  1. ', modelfile
  if write == 1:
    print '  2. ', gramfile  
    
  return  
         

def perplexity(modelfile, sentences, order=3, no_sos=0, no_eos=0, debug=0, \
  rmsent=1, msg=0):
  """
  get perplexity (ppl, ppl1) for text in sentences using SRILM
  """

  # print out the model file used
  if msg == 1: print 'model file:', modelfile

  # write sentence file (in the same directory with LM file)
  sentfile= os.path.join(os.path.dirname(modelfile), 'sentences.txt')
  with open(sentfile, 'wb') as f:
    if isinstance(sentences, str):
      f.writelines(sentences)
    elif isinstance(sentences, list):
      f.writelines('\n'.join(sentences))
    else:
      print 'sentences must be string or list of strings'
      return
  
  # make up command  
  command = ['ngram', '-lm', modelfile, '-ppl', sentfile, '-order', str(order)]
  if no_sos == 1: command += ['-no-sos']
  if no_eos == 1: command += ['-no-eos']
  if debug > 0: command += ['-debug', str(debug)]

  # run ngram in SRILM
  #print command
  result = subprocess.check_output(command)
  
  # example
  #result = subprocess.check_output(['ngram', '-lm', modelfile, '-ppl', \
  #  sentfile, '-order', str(order), '-no-sos', '-no-eos', '-debug', str(debug)])

  # match perplexity
  # pat = r'ppl[1]?= (\d+\.\d+)'
  pat = r'ppl[1]?= (\d+(\.\d+)?)'
  match = re.findall(pat, result)
  if len(match) > 0:
    match = [m[0] for m in match]
    ppl, ppl1 = [float(m) for m in match[-2:]]
  else:
    # hard coded a big number instead of inf if all words are OOVs
    ppl, ppl1 = 100000, 100000
  
  # remove temp sentence file
  if rmsent == 1: os.remove(sentfile)

  return ppl, ppl1, result   
