"""
Functions for data preparation
"""

import os
import fnmatch
import re
import codecs
from spellcheck import correct, simple_correct
from random import seed, shuffle
import numpy as np


def get_immediate_subdirectories(a_dir):
    """
    get immediate sub-directories
    """
    
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]
            
            
def get_files(folder, ext):
    """
    get files with specified extension
    """

    files = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, '*.'+ext):
            files.append(os.path.join(root, filename))
    return files

def validate_file(filename):
    """
    validate file in right format, such as no empty line
    """
    
    lines = file(filename).readlines()
    lines = [s.rstrip() for s in lines]
    lines = [s for s in lines if len(s)>0]
    with open(filename, 'wb') as f:
      f.writelines('\n'.join(lines))

def raw2sentence(filename):
    """
    convert raw text from file to sentences
    """
    
    source = codecs.open(filename, 'rb', encoding='utf-8')
    lines = source.readlines()       
        
    pat_timestamp = '[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]+'
    pat_comments = '[\s]?\[[A-Za-z|_]+\][\s|.]?'
    pat_playsound = '>>\s?'
    
    #pat_num = '\s([0-9]+)[.,!?; ]'
    #pat_num = r'(?<!\w)([\d.]+)(?!\w)'
    pat_num = r'(?<!\w)([\d]+(\.\d)?)(?!\w)'
    
    
    # process lines
    newlines = []
    for line in lines:
        
        # unicode -> string -> lowercase -> no return
        line = line.encode("utf-8").lower().strip()
        # remove '\xef\xbb\xbf' if needed
        try: 
            line = line.replace(u'\xef\xbb\xbf', '').encode()
        except:
            pass
        # deal with 'apostrophe'
        line = line.replace('\xe2\x80\x99s', ' is')
        line = line.replace('\xe2\x80\x99re', ' are')
        # deal with double quotes
        line = line.replace('\xe2\x80\x9c', '')
        line = line.replace('\xe2\x80\x9d', '')
        # remove prefix symbole for played sound
        line = re.sub(pat_playsound, '', line)
        # remove comments in []
        line = re.sub(pat_comments, ' ', line)

        # detect if it is empty
        flag_emptyline = 1 if line == '' or line.isspace() else 0
        
        # detect if it is digit only
        flag_digit = 1 if line.isdigit() else 0
        
        # detect if it is timestamp
        matches1 = re.findall(pat_timestamp, line)
        flag_timestamp = 1 if len(matches1) == 2 else 0
        
        # replace number by <num>
        matches = re.findall(pat_num, line)
        matches = [m[0] for m in matches]        
        matches.sort(key = lambda s: len(s))
        for m in reversed(matches):
            line = line.replace(m, '<num>')
        
        if not (flag_emptyline or flag_digit or flag_timestamp):
            newlines.append(line)
            
    # split into sentences
    sentences = []
    sent = ''
    for line in newlines:
        for ind, ch in enumerate(line):
            if (ch == '.' or ch == '?' or ch == '!') and \
            (ind == len(line)-1 or line[ind+1]==' '):
                flag_eos = 1
            else:
                flag_eos = 0
            if flag_eos == 1:
                sent += ch 
                sentences.append(sent.strip())
                sent = ''
            else:
                if ind == len(line)-1: 
                    sent += ch + ' '
                else:
                    sent += ch
     
    return sentences            
            
def spell_check(sentences):
    """
    spelling check for words in sentences
    WARNING: this version does not perform well 
    it makes more mistakes than corrections
    """
    
    newsentences = []
    newsentences0 = [] # keep a shallow copy of sentences without check
    for sent in sentences:
        words = re.findall(r"[\w'<>]+|[.,!?;]", sent)
        newwords = []
        newwords0 = []
        for word in words:
            newwords0.append(word)
            if word.isalpha():
                newwords.append(correct(word))
            else:
                newwords.append(word)
        newsentences.append(' '.join(newwords))
        newsentences0.append(' '. join(newwords0))
    return newsentences, newsentences0
    
def simple_check(sentences):
    """
    spelling check only to cover the words in the hacklist
    """
    
    newsentences = []
    for sent in sentences:
        words = re.findall(r"[\w'<>]+|[.,!?;]", sent)
        newwords = []
        for word in words:
            if word.isalpha():
                newwords.append(simple_correct(word))
            else:
                newwords.append(word)
        newsentences.append(' '.join(newwords))
    return newsentences  
      

def write_sentences(filename, sentences):
    """
    write sentences to file
    """

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wt') as f:
      f.writelines('\n'.join(sentences))    


def group_courses(dir_merge, courses, grouped_course, instructor, del_old=1):
    """
    group similar courses taught by the same instructor
    """
    
    files = [f for f in [c+'-'+instructor+'.txt' for c in courses]]
    
    merged_file = os.path.join(dir_merge, grouped_course + '-' + instructor + '.txt')
    with open(merged_file, "wb") as outfile:
        for f in files:
            individual_file = os.path.join(dir_merge, f)
            with open(individual_file, "rb") as infile:
                outfile.write(infile.read())
            if del_old == 1: os.remove(individual_file)
                

def del_course(dir_merge, course, instructor):
    """
    delete course text file that is too small
    """

    filename = os.path.join(dir_merge, course + '-' + instructor + '.txt')
    os.remove(filename)


def split_course(f, part, seed_int):
    """
    split text file into training, development and test
    based on partition and seed integer
    """

    # read in file and find the # of lines
    lines = file(f).readlines()
    nlines = len(lines)   

    # find the partitions by file indices (r[i]:r[r+1] for part. i)
    r = np.round(nlines * np.cumsum(np.array(part.values())))
    r = np.insert(r, 0, 0).astype(int)           

    # randomize indices with seed
    idx = list(range(nlines))
    seed(seed_int)
    shuffle(idx)
    
    # generate randomized lines
    lines_rand = [lines[i] for i in idx]
    
    # data split into 3 partitions
    for i in range(3):
        filename = os.path.splitext(f)[0].replace('stem', 'split', 1) + \
            '_' + part.keys()[i] + '_rand' + "{0:0>2}".format(seed_int) + '.txt'
        fo = open(filename, 'w')
        fo.writelines(lines_rand[r[i]:r[i+1]])
        fo.close()
