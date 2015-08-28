# -*- coding: utf-8 -*-
"""
Main script of spelling check of a single file
Zhenhao Ge, 2015-08-18
"""

import os, sys
pyver = sys.version_info[0]

# setup directories
directory = {}
path = r'D:/Dropbox/Work/Other/authorship_attribution/codes'
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
directory['data'] = os.path.join(directory['root'] + os.sep, 'data').replace('/', os.sep)  
directory['current'] = os.getcwd()    
if directory['current'] != directory['work']: os.chdir(directory['work'])
if pyver == 2: print 'current directory:', os.getcwd()
if pyver == 3: print('current directory:', os.getcwd())

# import spell check lib
from spellcheck import correct

# spelling check examples

# spelling check for single word
print correct('speling')
# use comprehension for multiple words
print [correct(s) for s in ['speling', 'adequiate']]
# it only allow one word per element, o.w. does not work
print [correct(s) for s in ['speling at', 'adequiate']] 



