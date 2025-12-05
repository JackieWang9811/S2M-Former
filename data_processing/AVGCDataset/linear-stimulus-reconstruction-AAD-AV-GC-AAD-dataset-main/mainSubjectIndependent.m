%% MAIN MMSE/LS-BASED DECODER WITH AVERAGING OF AUTOCORRELATION MATRICES AND RIDGE REGRESSION: SUBJECT-INDEPENDENT
% Author: Simon Geirnaert

clear; close all; clc;

%% Setup: parameters
params.subjects = [1,3,4,7,8,9,10,11,12,13,14,15,16]; % subjects to test (attend)
params.mode = 'leave-one-subject-out'; % 'leave-one-subject-out' or 'trained-on-KULeuven-AAD-2016' (trained on other dataset)
params.condition = 'all'; % condition to use: 'all' / 'NoVisuals' / 'FixedVideo' / 'MovingVideo' / 'MovingTargetNoise'
params.windowLengths = [600,300,120,60,30,20,10,5,2,1]; % different lengths decision windows to test (in seconds)
params.save = false; % save or not
params.saveName = 'saveName'; % name to save results with
params.datapath = ''; % fill in path to dataset

% decoder estimation
params.Leeg = [0,0.4]; % integration window ([0,250ms], post-stimulus)
params.regularization.name = 'lwcov'; % 'none': no regularization, 'lwcov': with ridge regression and heuristic regularization parameter determination

% preprocessing
params.preprocessing.passband = [1,9];
params.preprocessing.normalization = true;
params.preprocessing.normType = 'zscore'; % 'zscore' / 'froNorm' / 'center'

%% Initialization
acc = zeros(length(params.subjects),length(params.windowLengths)); % performance matrix: subject x decision window length
corrs = cell(length(params.subjects),length(params.windowLengths)); % all correlations with first column: attended correlation, second column: unattended correlation

%% Load or compute all correlation matrices
switch params.mode
    case 'leave-one-subject-out'
        fprintf('Computing covariance matrices \n');
        Rxx = []; RxyAtt = []; RxyUnatt = [];
        for s = 1:length(params.subjects)

            %% Load all data (EEG and speech envelopes, already preprocessed), labels and conditions and define parameters
            % load all data of subject s
            sb = params.subjects(s);
            [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGazeControl(sb,params.preprocessing,600,params.condition,params.datapath);
            nbTrials = length(attSpeaker);

            % extract attended speech envelopes
            attAudioTrials = zeros(size(audioTrials,1),size(audioTrials,3));
            unattAudioTrials = attAudioTrials;
            for tr = 1:nbTrials
                attAudioTrials(:,tr) = audioTrials(:,attSpeaker(tr),tr);
                unattAudioTrials(:,tr) = audioTrials(:,3-attSpeaker(tr),tr);
            end

            % create regression matrices
            [X,yAtt,yUnatt] = createRegressionMatrices(eegTrials,attAudioTrials,unattAudioTrials,params.Leeg,fs);
            X = reshape(permute(X,[2,1,3]),[size(X,2),size(X,1)*size(X,3)])'; yAtt = yAtt(:); yUnatt = yUnatt(:);

            % compute correlation matrices
            switch params.regularization.name
                case 'none'
                    RxxTemp = cov(X);
                case 'lwcov'
                    RxxTemp = lwcov(X);
            end
            RxyTempAtt = X'*yAtt; RxyTempUnatt = X'*yUnatt;
            Rxx = cat(3,Rxx,RxxTemp);
            RxyAtt = [RxyAtt,RxyTempAtt];
            RxyUnatt = [RxyUnatt,RxyTempUnatt];

        end
    case 'trained-on-KULeuven-AAD-2016'
        load dec-trainedOnDas.mat;
        dAtt = d;
end

%% Train and test decoders
for s = 1:length(params.subjects)
    fprintf('Testing subject %i/%i\n',s,length(params.subjects))

    %% Load test data (EEG and speech envelopes, already preprocessed), labels and conditions and define parameters
    % load all data of subject s
    testS = params.subjects(s);
    [eegTrials,audioTrials,attSpeaker,conditions,fs] = loadDataGazeControl(testS,params.preprocessing,600,params.condition,params.datapath);
    nbTrials = length(attSpeaker);

    % extract attended speech envelopes
    attAudioTrials = zeros(size(audioTrials,1),size(audioTrials,3));
    unattAudioTrials = attAudioTrials;
    for tr = 1:nbTrials
        attAudioTrials(:,tr) = audioTrials(:,attSpeaker(tr),tr);
        unattAudioTrials(:,tr) = audioTrials(:,3-attSpeaker(tr),tr);
    end

    if strcmp(params.mode,'leave-one-subject-out')
        %% Train decoder
        RxxTrain = mean(Rxx(:,:,setdiff(1:size(Rxx,3),s)),3);
        RxyTrainAtt = mean(RxyAtt(:,setdiff(1:size(Rxx,3),s)),2);
        RxyTrainUnatt = mean(RxyUnatt(:,setdiff(1:size(Rxx,3),s)),2);
        dAtt = RxxTrain\RxyTrainAtt;
    end

    %% Test decoder
    for wl = 1:length(params.windowLengths)
        %% create regression matrices per trial
        [Xtest,yAttTest,yUnattTest] = createRegressionMatrices(eegTrials,attAudioTrials,unattAudioTrials,params.Leeg,fs);

        %% Segment into smaller windows
        nbWindows = floor(size(Xtest,1)/(params.windowLengths(wl)*fs));
        Xtest = Xtest(1:nbWindows*params.windowLengths(wl)*fs,:,:);
        yAttTest = yAttTest(1:nbWindows*params.windowLengths(wl)*fs,:);
        yUnattTest = yUnattTest(1:nbWindows*params.windowLengths(wl)*fs,:);
        Xtest = segment(Xtest,params.windowLengths(wl)*fs);
        yAttTest = segmentize(yAttTest,'Segsize',params.windowLengths(wl)*fs); yAttTest = reshape(yAttTest,[size(yAttTest,1),size(yAttTest,2)*size(yAttTest,3)]);
        yUnattTest = segmentize(yUnattTest,'Segsize',params.windowLengths(wl)*fs); yUnattTest = reshape(yUnattTest,[size(yUnattTest,1),size(yUnattTest,2)*size(yUnattTest,3)]);

        %% feature extraction
        yPredAtt = squeeze(tmprod(Xtest,dAtt',2));
        rAttDec = [diag(corr(yPredAtt,yAttTest)),diag(corr(yPredAtt,yUnattTest))];
        corrs{s,wl} = [corrs{s,wl};rAttDec];
    end

    %% make decisions and compute AAD accuracy
    for wl = 1:length(params.windowLengths)
        acc(s,wl) = mean(corrs{s,wl}(:,1)>corrs{s,wl}(:,2));
    end
    disp(acc)

end
disp(squeeze(mean(acc,1)))

if params.save
    save(['results-subjectIndependent-',params.saveName],'acc','params','corrs');
end

%% Additional functions
function segmented = segment(X,segSize)
segmented = segmentize(X,'Segsize',segSize);
segmented = permute(segmented,[1,3,2,4]);
segmented = reshape(segmented,[size(segmented,1),size(segmented,2),size(segmented,3)*size(segmented,4)]);
end