% This script demonstrates how to import and align EEG and AUDIO data using
% the COCOHA Matlab Toolbox v0.5.0, found here: http://doi.org/10.5281/zenodo.1198430

EEGBASEPATH = './eeg';           % Find EEG files here
WAVBASEPATH = './audio';         % Find AUDIO wav files here 
MATBASEPATH = 'DTU_vanilla_dataset+preproc+128';               % Save preprocessed data files here

if ~exist(MATBASEPATH, 'dir')
    mkdir(MATBASEPATH);
    fprintf('Directory %s created.\n', MATBASEPATH);
end

for ss = 1:18
    clear data
    fprintf('Processing subject: %s\n', num2str(ss));
    
    %% Load data
    load(fullfile(EEGBASEPATH,['S' num2str(ss) '.mat']))
    
    %% Assign L/R events
    events_of_interest = expinfo.attend_mf; % Did the subject listen to the male or female speaker?   
    
    events = cat(1,data.event.eeg.value{:}); 
    if sum(events(1:2:end)-expinfo.trigger)==0; fprintf('Matching triggers \n'); end
    for ii = 1:length(events_of_interest); data.event.eeg.value{2*(ii-1)+1} = events_of_interest(ii);  end
    data.cfg = [];
    
    %% Line noise filtering 50 Hz
    cfg = [];
    cfg.eeg.smooth = data.fsample.eeg/50;
    data = co_preprocessing(cfg,data);

    %% Downsample
    cfg = [];
    cfg.eeg.newfs = 128;
    data = co_resampledata(cfg,data);

    %% Initial filtering
    cfg = [];
    cfg.eeg.detrend = 1;    
    cfg.eeg.hpfilter = 'yes';
    cfg.eeg.hpfilttype = 'butter'; 
    cfg.eeg.hpfiltord = 2; 
    cfg.eeg.hpfiltdir = 'onepass';
    cfg.eeg.hpfreq = 0.1;
    data = co_preprocessing(cfg,data);

    %% Create EOG bipolar channels
    cfg = [];
    cfg.eeg.channels = {'EXG3','EXG5'};
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = {'EXG5'};
    data_veog = co_preprocessing(cfg,data);

    cfg = [];
    cfg.eeg.channels = {'EXG4','EXG7'};
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = {'EXG7'};
    data_heog = co_preprocessing(cfg,data);

    %% Remove original EOG and unused channels from data and average reference
    cfg = [];
    cfg.eeg.channels = {'all','-EXG3','-EXG4','-EXG5','-EXG6','-EXG7','-EXG8','-Status'};
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = 'all';
    data = co_preprocessing(cfg,data);

    %% Append bipolar EOG channels to data
    cfg = [];
    cfg.eeg.dim = 'chan';
    data = co_appenddata(cfg, data, data_veog, data_heog);

    %% Denoising
    cfg = [];
    cfg.eeg.eog.channels = {'EXG3','EXG4'};
    data = co_denoise(cfg,data);

    %% Remove EOG channels
    cfg = [];
    cfg.eeg.dim = 'chan';
    cfg.eeg.select = {'all','-EXG3','-EXG4'};
    data = co_selectdim(cfg,data);

    %% Average reference
    cfg = [];
    cfg.eeg.reref = 'yes';
    cfg.eeg.refchannel = 'all';
    data = co_preprocessing(cfg,data);

    %% Select events corresponding to attended talker
    cfg = [];
    cfg.eeg.event = {1,2};
    data = co_selectevent(cfg,data);

    %% Split continuous data into trials
    cfg = [];
    cfg.eeg.splitsample = data.event.eeg.sample;
    data = co_splitdata(cfg,data);

    %% Split data into cells
    data_cell = cell(1,length(data.eeg)-1);
%     data_noise = cell(1,length(data.eeg));

    for ii = 1:length(data_cell)
        cfg = [];
        cfg.eeg.cell = ii+1;    % Ignore first cell (pre-stim)
        data_cell{ii} = co_selectcell(cfg,data);
    end
    
%     data_noise{1} = data.eeg{1};
    data = data_cell;
    clear data_cell

    %% Add attended and unattended audio and extract envelopes
    prefix = '';  
    story_names = [expinfo.wavfile_male expinfo.wavfile_female];
    for ii = 1:length(data)
        cfg = [];
        cfg.wavA.dataset = fullfile(WAVBASEPATH,[story_names{ii,data{ii}.event.eeg.value{1}}]);
        data{ii} = co_preprocessing(cfg,data{ii});
        cfg = []; cfg.wavA.dim = 'time'; data{ii} = co_auditoryfilterbank(cfg,data{ii});
        data{ii}.wavA{1} = abs(data{ii}.wavA{1});
        data{ii}.wavA{1} = data{ii}.wavA{1}.^.3;
    
        unattix = mod(data{ii}.event.eeg.value{1},2)+1;
        if ~isempty(~story_names{ii,unattix})
            cfg = [];
            cfg.wavB.dataset = fullfile(WAVBASEPATH,[story_names{ii,unattix}]);
            data{ii} = co_preprocessing(cfg,data{ii});
            cfg = []; cfg.wavB.dim = 'time'; data{ii} = co_auditoryfilterbank(cfg,data{ii});
            data{ii}.wavB{1} = abs(data{ii}.wavB{1});
            data{ii}.wavB{1} = data{ii}.wavB{1}.^.3;
        end

        % Downsample to AUDIO to EEG sampling rate
        cfg = [];
        cfg.wavA.newfs = data{ii}.fsample.eeg;
        if isfield(data{ii},'wavB'); cfg.wavB.newfs = cfg.wavA.newfs; end;
        data{ii} = co_resampledata(cfg,data{ii});
        
        % Average AUDIO frequency channels
        cfg = []; cfg.wavA.dimlabel = 'freq'; cfg.wavA.dim = {'freq'};
        if isfield(data{ii},'wavB'); cfg.wavB.dimlabel = 'freq'; cfg.wavB.dim={'freq'}; end;
        data{ii} = co_dimaverage(cfg,data{ii});
        data{ii} = co_squeeze(cfg,data{ii});
    end
    
    %% Remove single-talker trials with no unattended talker
    keep_ix = false(1,length(data));
    for ii = 1:length(data)
        if isfield(data{ii},'wavB'); keep_ix(ii) = true; end;
    end
    data = data(keep_ix);
%     data_noise = data_noise(keep_ix);

    %% Trim trials to be the same length
    for ii = 1:length(data)
        try
       cfgtmp = []; 
       cfgtmp.eeg.dim = 'time'; eeglen = co_dimlen(cfgtmp,data{ii});
        cfgtmp = []; cfgtmp.wavA.dim = 'time'; wavAlen = co_dimlen(cfgtmp,data{ii});
        cfgtmp = []; cfgtmp.wavB.dim = 'time'; wavBlen = co_dimlen(cfgtmp,data{ii});
        minlen = min([eeglen,wavAlen,wavBlen]);
        catch
            minlen = min([size(data{1}.eeg{1},1),size(data{1}.wavA{1},1),size(data{1}.wavB{1},1)]);
        end
        
%         data_noise{ii+1} = data{ii}.eeg{1}(minlen+1:end,:);
        
        cfg = [];
        cfg.eeg.dim = 'time'; cfg.wavA.dim = 'time'; cfg.wavB.dim = 'time';
        cfg.eeg.select = 1:minlen; cfg.wavA.select = 1:minlen; cfg.wavB.select = 1:minlen;
        data{ii} = co_selectdim(cfg,data{ii});
    end

    %% Append data cells as trials
    cfg = [];
    cfg.eeg = [];
    cfg.wavA = [];
    cfg.wavB = [];
    data = co_appenddata(cfg,data);
    
    %% Save data
    fprintf('Saving data\n')
    save(fullfile(MATBASEPATH,['S' num2str(ss) '_data_preproc.mat']),'data');
%     save(fullfile(MATBASEPATH,['S' num2str(ss) '_noise_preproc.mat']),'data_noise');
end