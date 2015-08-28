function [filename,filesize] = getfile(path,ext)
% Extract files from the specified direcotry (path)
%
% Inputs:
% path - directory containing the files to be processed
% ext - extension name
%
% Output:
% file - list of files
%
% Author: Roger Ge, roger.ge@inin.com
% Date created: 2012-07-11

if nargin < 2, ext = ''; end

tmp = struct2cell(dir(path));
filename = tmp(1,3:end);
filesize = tmp(3,3:end);

% select files in current folder only (not recursively)
fileFlag = cell2mat(tmp(4,3:end))==0;
filename = filename(fileFlag);
filesize = filesize(fileFlag);

if strcmp(ext,'')
    
    filename = filename';
    
else
    
    extension = cellfun(@(v) v(end-length(ext):end), filename, ...
        'UniformOutput',0);
    flag = strcmp(extension,['.',ext]);
    filename = filename(flag==1)';
    filesize = filesize(flag==1)';

end