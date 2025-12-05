function [X,yAtt,yUnatt] = createRegressionMatrices(eegTrials,attAudioTrials,unAttAudioTrials,L,fs)

%% Initialization
nbTrials = size(eegTrials,3);
nbChan = size(eegTrials,2);
T = size(eegTrials,1);
yAtt = attAudioTrials;
yUnatt = unAttAudioTrials;
zerosBefore = abs(min(round(L(1)*fs),0));
zerosAfter = abs(max(round(L(2)*fs),0));
nbLags = zerosAfter-round(L(1)*fs)+1;
X = zeros(T,nbLags*nbChan,nbTrials);

%% Construct data matrices
for tr = 1:nbTrials
    Xtemp = eegTrials(:,:,tr);
    % Append Toeplitz matrices
    for ch = 1:size(Xtemp,2)
        fC = [Xtemp(:,ch);zeros(zerosAfter,1)];
        fC = fC(end-T+1:end);
        fR = [Xtemp(zerosAfter+1:-1:(max(round(L(1)*fs),0)+1),ch)',zeros(1,max(zerosBefore,0))];
        X(:,(ch-1)*nbLags+1:ch*nbLags,tr) = toeplitz(fC,fR);
    end
end

end