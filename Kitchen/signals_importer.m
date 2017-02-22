function [LIGHT1,PIR1,TEMP1,HUMID1,LIGHT2,PIR2,TEMP2,HUMID2,LIGHT3,PIR3,TEMP3,HUMID3,LIGHT4,PIR4,TEMP4,HUMID4,LIGHT5,PIR5,TEMP5,HUMID5,LIGHT6,PIR6,TEMP6,HUMID6,X,Y,THETA,TARGET] = signals_importer(filename, startRow, endRow)
%IMPORTFILE Import numeric data from a text file as column vectors.
%   [LIGHT1,PIR1,TEMP1,HUMID1,LIGHT2,PIR2,TEMP2,HUMID2,LIGHT3,PIR3,TEMP3,HUMID3,LIGHT4,PIR4,TEMP4,HUMID4,LIGHT5,PIR5,TEMP5,HUMID5,LIGHT6,PIR6,TEMP6,HUMID6,X,Y,THETA,TARGET]
%   = IMPORTFILE(FILENAME) Reads data from text file FILENAME for the
%   default selection.
%
%   [LIGHT1,PIR1,TEMP1,HUMID1,LIGHT2,PIR2,TEMP2,HUMID2,LIGHT3,PIR3,TEMP3,HUMID3,LIGHT4,PIR4,TEMP4,HUMID4,LIGHT5,PIR5,TEMP5,HUMID5,LIGHT6,PIR6,TEMP6,HUMID6,X,Y,THETA,TARGET]
%   = IMPORTFILE(FILENAME, STARTROW, ENDROW) Reads data from rows STARTROW
%   through ENDROW of text file FILENAME.
%
% Example:
%   [LIGHT1,PIR1,TEMP1,HUMID1,LIGHT2,PIR2,TEMP2,HUMID2,LIGHT3,PIR3,TEMP3,HUMID3,LIGHT4,PIR4,TEMP4,HUMID4,LIGHT5,PIR5,TEMP5,HUMID5,LIGHT6,PIR6,TEMP6,HUMID6,X,Y,THETA,TARGET] = importfile('kitchen1.csv',2, 51);
%
%    See also TEXTSCAN.

% Auto-generated by MATLAB on 2016/11/14 10:15:22

%% Initialize variables.
delimiter = ',';
if nargin<=2
    startRow = 2;
    endRow = inf;
end

%% Format string for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
%	column4: double (%f)
%   column5: double (%f)
%	column6: double (%f)
%   column7: double (%f)
%	column8: double (%f)
%   column9: double (%f)
%	column10: double (%f)
%   column11: double (%f)
%	column12: double (%f)
%   column13: double (%f)
%	column14: double (%f)
%   column15: double (%f)
%	column16: double (%f)
%   column17: double (%f)
%	column18: double (%f)
%   column19: double (%f)
%	column20: double (%f)
%   column21: double (%f)
%	column22: double (%f)
%   column23: double (%f)
%	column24: double (%f)
%   column25: double (%f)
%	column26: double (%f)
%   column27: double (%f)
%	column28: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
LIGHT1 = dataArray{:, 1};
PIR1 = dataArray{:, 2};
TEMP1 = dataArray{:, 3};
HUMID1 = dataArray{:, 4};
LIGHT2 = dataArray{:, 5};
PIR2 = dataArray{:, 6};
TEMP2 = dataArray{:, 7};
HUMID2 = dataArray{:, 8};
LIGHT3 = dataArray{:, 9};
PIR3 = dataArray{:, 10};
TEMP3 = dataArray{:, 11};
HUMID3 = dataArray{:, 12};
LIGHT4 = dataArray{:, 13};
PIR4 = dataArray{:, 14};
TEMP4 = dataArray{:, 15};
HUMID4 = dataArray{:, 16};
LIGHT5 = dataArray{:, 17};
PIR5 = dataArray{:, 18};
TEMP5 = dataArray{:, 19};
HUMID5 = dataArray{:, 20};
LIGHT6 = dataArray{:, 21};
PIR6 = dataArray{:, 22};
TEMP6 = dataArray{:, 23};
HUMID6 = dataArray{:, 24};
X = dataArray{:, 25};
Y = dataArray{:, 26};
THETA = dataArray{:, 27};
TARGET = dataArray{:, 28};


