# -*- coding: utf-8 -*-
"""
Main script for data prepation
    1. convert raw text to sentence-wise text (with word/punctuation seperation
       and simple spelling check)
    2. merge course-instructor files
    3. randomly split into train/dev/test (8:1:1)
    
Zhenhao (Roger) Ge, 2015-08-18    
"""

# import public modules
import os, sys, subprocess
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
directory['data'] = os.path.join(directory['root'] + os.sep, 'data').replace('/', os.sep)  
directory['current'] = os.getcwd()    
if directory['current'] != directory['work']: os.chdir(directory['work'])
if pyver == 2: print 'current directory:', os.getcwd()
if pyver == 3: print('current directory:', os.getcwd())

# import local modules    
from dataprep import get_immediate_subdirectories, get_files, validate_file
from dataprep import raw2sentence, simple_check, write_sentences
from dataprep import group_courses, del_course, split_course    

# find course-instructor folders
directory['raw'] = os.path.join(directory['data'], 'raw')
directory['sent'] = os.path.join(directory['data'], 'sentence') 
directory['merge'] = os.path.join(directory['data'], 'merge') 
directory['stem'] = os.path.join(directory['data'], 'stem') 
directory['split'] = os.path.join(directory['data'], 'split') 
folders = get_immediate_subdirectories(directory['raw'])            

# double-loop to convert each text file in each course-instructor folder from
# raw format to sentence format
for idx1, folder in enumerate(folders):
    
    if pyver == 2: print 'processing (folder ' + str(idx1) + '):', folder
    if pyver == 3: print('processing (folder ' + str(idx1) + '):', folder) 
    
    # find files in each folder
    files = sorted(get_files(os.path.join(directory['raw'], folder), 'txt'))

    for idx2, filename in enumerate(files):
        
        #if pyver == 2: print 'processing (file ' + str(idx2) + '):', filename
        #if pyver == 3: print('processing (file ' + str(idx2) + '):', filename)    
        
        # convert raw text to sentence text
        sentences = raw2sentence(filename)
        
        # perform simple spelling check (only correct words in hack list)
        newsentences = simple_check(sentences)
             
        # write the sentence file with simple spelling check
        filename2 = filename.replace(os.sep+'raw'+os.sep, os.sep+'sentence'+os.sep)
        write_sentences(filename2, newsentences) 
        
# merge files from the same course and instructor
for idx1, folder in enumerate(folders):
    
    # find files in each folder
    files = sorted(get_files(os.path.join(directory['sent'], folder), 'txt'))
    
    #course_name, instructor_name = folder.split('-')
    
    merged_file = os.path.join(directory['merge'], folder+'.txt')
    with open(merged_file, "wb") as outfile:
        for f in files:
            with open(f, "rb") as infile: 
                outfile.write(infile.read()+'\n')
                           
# group some similar courses taught by the same instructor
courses = ['PatternDiscovery', 'ClusterAnalysis']
grouped_course = 'DataMining1'
instructor = 'JiaweiHan'
if pyver == 2: print 'grouping', ', '.join(courses), 'as', grouped_course, '...'
if pyver == 3: print('grouping', ', '.join(courses), 'as', grouped_course, '...')              
group_courses(directory['merge'], courses, grouped_course, instructor)

courses = ['TextRetrieval', 'TextAnalytics']
grouped_course = 'DataMining2'
instructor = 'ChengxiangZhai'
if pyver == 2: print 'grouping', ', '.join(courses), 'as', grouped_course, '...'
if pyver == 3: print('grouping', ', '.join(courses), 'as', grouped_course, '...')              
group_courses(directory['merge'], courses, grouped_course, instructor)

# delete merged text file that is too small
course = 'DataVisualization'
instructor = 'JohnHart'
if pyver == 2: print 'deleting', course, '...'
if pyver == 3: print('deleting', course, '...')
del_course(directory['merge'], course, instructor)

course = 'Acoustics1'
instructor = 'YangHannKim'
if pyver == 2: print 'deleting', course, '...'
if pyver == 3: print('deleting', course, '...')
del_course(directory['merge'], course, instructor)

# stem mergerd file (after grouping similar courses)
files = sorted(get_files(directory['merge'], 'txt'))
for idx, filename in enumerate(files):
    
    if pyver == 2: print 'processing (file ' + str(idx) + '):', filename
    if pyver == 3: print('processing (file ' + str(idx) + '):', filename)
    
    # validate merged file before stem it
    validate_file(filename)
    
    # stem the merged file and save in the stemmed file
    filename2 = filename.replace(os.sep+'merge'+os.sep, os.sep+'stem'+os.sep)          
    with open(filename2, 'wb') as f:        
        f.write(subprocess.check_output(["python", "porterStemmer.py", filename]))

# split processed text file into training, developement and test
if pyver == 2: print 'spliting files ...'
if pyver == 3: print('spliting files ...')
files = sorted(get_files(directory['stem'], 'txt'))
part = {'train':0.8, 'valid': 0.1, 'test': 0.1}
num_seeds = 10
for f in files:
    for seed_int in range(num_seeds):
        split_course(f, part, seed_int)
         