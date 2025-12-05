%% MAIN SUBJECT-SPECIFIC LINEAR STIMULUS DECODING ON AV-GC-AAD DATASET
% Author: Simon Geirnaert

clear; close all; clc;

%% Setup: parameters
params.subjects = [1,3,4,7,8,9,10,11,12,13,14,15,16]; % subjects to test (attend) [note: leave out subject 14 for static video, subjects 1 and 3 for moving target + noise]
params.subjects = [1]; % subjects to test (attend) [note: leave out subject 14 for static video, subjects 1 and 3 for moving target + noise]

params.condition = 'NoVisuals'; % condition to use: 'all' / 'NoVisuals' / 'StaticVideo' / 'MovingVideo' / 'MovingTargetNoise'
params.save = true; % save or not
params.saveName = 'saveName'; % name to save results with
params.datapath = 'D:\4th_SCI_work_SNN_AAD\OpenAAD\data_processing\AVGCDataset\linear-stimulus-reconstruction-AAD-AV-GC-AAD-dataset-main'; % fill in path to dataset
% decoder estimation

% cross-validation
params.cv.type = 'per-trial'; % 'folds' / 'LOO' / 'per-trial' / 'per-condition'

% preprocessing
params.preprocessing.normalization = true;
params.preprocessing.normType = 'zscore'; % 'zscore' / 'froNorm' / 'center'


%% Loop over subjects
for s = 1:length(params.subjects)
    
    %% Load all data (EEG and speech envelopes, already preprocessed), labels and conditions and define parameters
    % load all data of subject s
    testS = params.subjects(s);
    switch params.cv.type
        case {'per-trial','per-condition'}
            [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGaze_Preprocessed(testS,params.preprocessing,600,params.condition,params.datapath);
        otherwise
            [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGaze_Preprocessed(testS,params.preprocessing,max(params.windowLengths),params.condition,params.datapath);
    end
end
