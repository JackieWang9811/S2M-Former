function [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGazeControl(subject,preprocessing,segmentLength,condition,datapath)
% LOADDATAGAZECONTROL Load the EEG and attended speaker of a given subject and
% condition. The trials are preprocessed with the given parameters in
% preprocessing (filtering and normalization) and split in segments of a given segmentLength.

%% load correct data
data = load([datapath,'\2024-AV-GC-AAD-sub',sprintf('%02d',subject),'_preprocessed.mat']);

nbTrials = floor(size(data.data{1},1)/(segmentLength*data.fs));
fs = data.fs;

%% split into shorter segments
eegTrials = []; audioTrials = []; conditions = [];

for tr = 1:length(data.data)
    data.eegTrials{tr} = double(data.data{tr}(1:nbTrials*segmentLength*data.fs,:));
    data.audioTrials{tr} = [double(data.stimulus.attendedEnvelopes{tr}(1:nbTrials*segmentLength*data.fs)),double(data.stimulus.unattendedEnvelopes{tr}(1:nbTrials*segmentLength*data.fs))];
    eegTemp = reshape(data.eegTrials{tr},[segmentLength*data.fs,nbTrials,size(data.eegTrials{tr},2)]);
    audioTemp = reshape(data.audioTrials{tr},[segmentLength*data.fs,nbTrials,size(data.audioTrials{tr},2)]);
    eegTrials = cat(2,eegTrials,eegTemp);
    audioTrials = cat(2,audioTrials,audioTemp);

    % code conditions
    if contains(data.conditionID{tr},'NoVisuals')
        conditions = [conditions;1*ones(nbTrials,1)];
    elseif contains(data.conditionID{tr},'StaticVideo')
        conditions = [conditions;2*ones(nbTrials,1)];
    elseif contains(data.conditionID{tr},'MovingVideo')
        conditions = [conditions;3*ones(nbTrials,1)];
    elseif contains(data.conditionID{tr},'MovingTargetNoise')
        conditions = [conditions;4*ones(nbTrials,1)];
    end
end
eegTrials = permute(eegTrials,[1,3,2]); audioTrials = permute(audioTrials,[1,3,2]);
attSpeaker = ones(size(eegTrials,3),1);

%% select correct conditions
switch condition
    case 'all'
    case 'NoVisuals'
        eegTrials = eegTrials(:,:,conditions==1); audioTrials = audioTrials(:,:,conditions==1); attSpeaker = attSpeaker(conditions==1); conditions = conditions(conditions==1);
    case 'StaticVideo'
        eegTrials = eegTrials(:,:,conditions==2); audioTrials = audioTrials(:,:,conditions==2); attSpeaker = attSpeaker(conditions==2); conditions = conditions(conditions==2);
    case 'MovingVideo'
        eegTrials = eegTrials(:,:,conditions==3); audioTrials = audioTrials(:,:,conditions==3); attSpeaker = attSpeaker(conditions==3); conditions = conditions(conditions==3);
    case 'MovingTargetNoise'
        eegTrials = eegTrials(:,:,conditions==4); audioTrials = audioTrials(:,:,conditions==4); attSpeaker = attSpeaker(conditions==4); conditions = conditions(conditions==4);
end
nbTrials = size(eegTrials,3);

%% preprocessing
% select only EEG
eegTrials = eegTrials(:,1:64,:);

% filtering
bpFilter = designfilt('bandpassiir','FilterOrder',4,'HalfPowerFrequency1',preprocessing.passband(1),'HalfPowerFrequency2',preprocessing.passband(2),'SampleRate',fs);
eegTrials = filtfilt(bpFilter,eegTrials);
audioTrials = filtfilt(bpFilter,audioTrials);

% downsampling
% fsPos = 2*preprocessing.passband(2):fs;
% fsNew = fsPos(find(mod(fs,2*preprocessing.passband(2):fs)==0,1,'first'));
fsNew = 20;
eegTrials = resample(eegTrials,fsNew,fs);
audioTrials = resample(audioTrials,fsNew,fs);
fs = fsNew;

% normalization
if preprocessing.normalization
    if strcmp(preprocessing.normType,'zscore')
        eegTrials = zscore(eegTrials,[],1);
    elseif strcmp(preprocessing.normType,'center')
        eegTrials = eegTrials-mean(eegTrials,1);
        audioTrials = audioTrials-mean(audioTrials,1);
    elseif strcmp(preprocessing.normType,'froNorm')
        for tr = 1:nbTrials
            eegTrials(:,:,tr) = eegTrials(:,:,tr)./norm(eegTrials(:,:,tr),'fro')*size(eegTrials,2);
        end
        eegTrials = eegTrials-mean(eegTrials,1);
    end
end
audioTrials = zscore(audioTrials,[],1);

end