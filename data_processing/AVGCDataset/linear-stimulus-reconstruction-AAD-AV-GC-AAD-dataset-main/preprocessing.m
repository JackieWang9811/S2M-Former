%% MAIN SUBJECT-SPECIFIC LINEAR STIMULUS DECODING ON AV-GC-AAD DATASET
% Author: Simon Geirnaert

clear; close all; clc;

%% Setup: parameters
params.subjects = [1,3,4,7,8,9,10,11,12,13,14,15,16]; % subjects to test (attend) [note: leave out subject 14 for static video, subjects 1 and 3 for moving target + noise]
params.condition = 'NoVisuals'; % condition to use: 'all' / 'NoVisuals' / 'StaticVideo' / 'MovingVideo' / 'MovingTargetNoise'
params.windowLengths = [600,300,120,60,30,20,10,5,2,1]; % different lengths decision windows to test (in seconds)
params.save = true; % save or not
params.saveName = 'saveName'; % name to save results with
params.datapath = 'D:\4th_SCI_work_SNN_AAD\OpenAAD\data_processing\AVGCDataset\linear-stimulus-reconstruction-AAD-AV-GC-AAD-dataset-main'; % fill in path to dataset


% preprocessing
params.preprocessing.normalization = true;
params.preprocessing.normType = 'zscore'; % 'zscore' / 'froNorm' / 'center'


%% Loop over subjects
for s = 1:length(params.subjects)
    
    %% Load all data (EEG and speech envelopes, already preprocessed), labels and conditions and define parameters
    [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGazeControl(testS,params.preprocessing,600,params.condition,params.datapath);
    end

    % determine outer CV folds
    nbTrials = length(attSpeaker);
    switch params.cv.type
        case 'folds'
            cvOuter = cvpartition(nbTrials,'KFold',params.cv.nbFoldsOuter);
        case 'LOO'
            params.cv.nbFoldsOuter = nbTrials;
            cvOuter = eye(nbTrials);
        case 'per-trial'
            params.cv.nbFoldsOuter = nbTrials;
            cvOuter = eye(nbTrials);
        case 'per-condition'
            params.cv.nbFoldsOuter = length(unique(conditions));
            cvOuter = zeros(nbTrials,params.cv.nbFoldsOuter);
            for c = 1:length(unique(conditions))
                cvOuter(conditions==c,c) = 1;
            end
    end
    
    % extract attended speech envelopes
    attAudioTrials = zeros(size(audioTrials,1),size(audioTrials,3));
    unattAudioTrials = attAudioTrials;
    for tr = 1:nbTrials
        attAudioTrials(:,tr) = audioTrials(:,attSpeaker(tr),tr);
        unattAudioTrials(:,tr) = audioTrials(:,3-attSpeaker(tr),tr);
    end
    trialLength = size(eegTrials,1);
    
    %% Construct training and test folds   
    cv = cell(params.cv.nbFoldsOuter,1); % create crossvalidation structure, per fold
    for fold = 1:params.cv.nbFoldsOuter
        if strcmp(params.cv.type,'LOO') || strcmp(params.cv.type,'per-trial') || strcmp(params.cv.type,'per-condition')
            cv{fold}.training.eeg = eegTrials(:,:,logical(1-cvOuter(:,fold)));
            cv{fold}.training.attAudio = attAudioTrials(:,logical(1-cvOuter(:,fold)));
            cv{fold}.training.unattAudio = unattAudioTrials(:,logical(1-cvOuter(:,fold)));
            cv{fold}.test.eeg = eegTrials(:,:,logical(cvOuter(:,fold)));
            cv{fold}.test.attAudio = attAudioTrials(:,logical(cvOuter(:,fold)));
            cv{fold}.test.unattAudio = unattAudioTrials(:,logical(cvOuter(:,fold)));
            cv{fold}.test.label = attSpeaker(logical(cvOuter(:,fold)));
            cv{fold}.test.videoCondition = conditions(logical(cvOuter(:,fold)));
        else
            cv{fold}.training.eeg = eegTrials(:,:,cvOuter.training(fold));
            cv{fold}.training.attAudio = attAudioTrials(:,cvOuter.training(fold));
            cv{fold}.training.unattAudio = unattAudioTrials(:,cvOuter.training(fold));
            cv{fold}.test.eeg = eegTrials(:,:,cvOuter.test(fold));
            cv{fold}.test.attAudio = attAudioTrials(:,cvOuter.test(fold));
            cv{fold}.test.unattAudio = unattAudioTrials(:,cvOuter.test(fold));
            cv{fold}.test.label = attSpeaker(cvOuter.test(fold));
            cv{fold}.test.videoCondition = conditions(cvOuter.test(fold));
        end
    end

    %% Compute decoders per fold and window length (for averaging decoders) and cross-validate regularization constant
    parfor fold = 1:params.cv.nbFoldsOuter
        fprintf('Subject %i/%i, training: fold %i/%i\n',s,length(params.subjects),fold,params.cv.nbFoldsOuter)

        % create regression matrices per trial
        [X,yAtt,yUnatt] = createRegressionMatrices(cv{fold}.training.eeg,cv{fold}.training.attAudio,cv{fold}.training.unattAudio,params.Leeg,fs);
        Xtrain = reshape(permute(X,[1,3,2]),size(X,1)*size(X,3),size(X,2));
        yAttTrain = yAtt(:); yUnattTrain = yUnatt(:);

        % compute decoder
        cv{fold}.dAtt = trainDecoder(Xtrain,yAttTrain,params.regularization);
   end

    %% Apply optimal decoder per window length on the test set
    for fold = 1:params.cv.nbFoldsOuter
        fprintf('Subject %i/%i, testing: fold %i/%i\n',s,length(params.subjects),fold,params.cv.nbFoldsOuter)

        for wl = 1:length(params.windowLengths)
            %% create regression matrices per trial
            [Xtest,yAttTest,yUnattTest] = createRegressionMatrices(cv{fold}.test.eeg,cv{fold}.test.attAudio,cv{fold}.test.unattAudio,params.Leeg,fs);

            %% Segment into smaller windows
            nbWindows = floor(size(Xtest,1)/(params.windowLengths(wl)*fs));
            Xtest = Xtest(1:nbWindows*params.windowLengths(wl)*fs,:,:);
            yAttTest = yAttTest(1:nbWindows*params.windowLengths(wl)*fs,:);
            yUnattTest = yUnattTest(1:nbWindows*params.windowLengths(wl)*fs,:);
            Xtest = segment(Xtest,params.windowLengths(wl)*fs);
            yAttTest = segmentize(yAttTest,'Segsize',params.windowLengths(wl)*fs); yAttTest = reshape(yAttTest,[size(yAttTest,1),size(yAttTest,2)*size(yAttTest,3)]);
            yUnattTest = segmentize(yUnattTest,'Segsize',params.windowLengths(wl)*fs); yUnattTest = reshape(yUnattTest,[size(yUnattTest,1),size(yUnattTest,2)*size(yUnattTest,3)]);
            
            %% feature extraction
            yPredAtt = squeeze(tmprod(Xtest,cv{fold}.dAtt',2));
            rAttDec = [diag(corr(yPredAtt,yAttTest)),diag(corr(yPredAtt,yUnattTest))];
            corrs{s,wl} = [corrs{s,wl};rAttDec];

            %% save conditions
            cond{s,wl} = [cond{s,wl};repelem(cv{fold}.test.videoCondition,nbWindows,1)];
        end
    end
    
    %% make decisions and compute AAD accuracy
    for wl = 1:length(params.windowLengths)
        acc(s,wl) = mean(corrs{s,wl}(:,1)>corrs{s,wl}(:,2));
    end
    disp(squeeze(acc))
end
disp(squeeze(mean(acc,1)))

if params.save
    save(['results-subjectSpecific-',params.saveName],'acc','params','corrs','conds');
end

%% Additional functions
function segmented = segment(X,segSize)
    segmented = segmentize(X,'Segsize',segSize);
    segmented = permute(segmented,[1,3,2,4]);
    segmented = reshape(segmented,[size(segmented,1),size(segmented,2),size(segmented,3)*size(segmented,4)]);
end