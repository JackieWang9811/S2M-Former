function [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGazeControl(subject,preprocessing,segmentLength,condition,datapath)

%% load correct data
data = load([datapath,'\2024-AV-GC-AAD-sub',sprintf('%02d',subject),'_preprocessed.mat']);

nbTrials = floor(size(data.data{1},1)/(segmentLength*data.fs));
fs = data.fs;

%% split into shorter segments
eegTrials = []; audioTrials = []; conditions = [];

for tr = 1:length(data.data)
    data.eegTrials{tr} = double(data.data{tr}(1:nbTrials*segmentLength*data.fs,:));
    data.eegTrials{tr} = data.eegTrials{tr}(:, 1:64);
    % normalization
    if preprocessing.normalization
        if strcmp(preprocessing.normType,'zscore')
            data.eegTrials{tr} = zscore(data.eegTrials{tr},[],1);
        end
    end
end

save([datapath, '\2024-AV-GC-AAD-sub',sprintf('%02d',subject),'_preprocessed_zscore.mat']);

end