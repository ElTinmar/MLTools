function [decoding,confusion] = MLD(A)
% maximum likelihood decoder
% A is a matrix of size n_neurons x n_stim x n_trials
% one way to obtain A is to average neuronal activity
% over a time window after the onset of each stimulus 

    [n_neurons,n_stim,n_trials] = size(A);
    
    % leave-one-out  
    P = NaN(n_neurons,n_stim,n_stim,n_trials);
    for trial = 1:n_trials
        loo_ind = setdiff(1:n_trials,trial);
        for stim_type1 = 1:n_stim
            for stim_type2 = 1:n_stim
                P(:,stim_type1,stim_type2,trial) = log(kde(squeeze(A(:,stim_type2,loo_ind)),A(:,stim_type1,trial)));
            end
        end
    end
    
    % IID 
    LP_iid = squeeze(sum(P,1)); % n_stim (test) x n_stim (distrib. est.) x n_trials 
    [~,argmax] = max(LP_iid,[],2,'omitnan');
    decoding = squeeze(argmax);
    confusion = zeros(n_stim,n_stim);
    for i = 1:n_stim
        for j = 1:n_trials
            confusion(i,decoding(i,j)) = confusion(i,decoding(i,j)) + 1;
        end
    end
    
    function [f,h] = kde(x,xq)
    % 1D kernel density estimation on rows of 2D matrix x
    % xq : points where to evaluate the density, 1D column vector
    % x  : 2D matrix of samples 

        % Bandwidth estimation: Scott's rule
        N = size(x,2);
        D = 1; % dimension
        sig = mad(x,1,2) / 0.6745; % robust estimation of spread
        h = sig * (4/((D+2)*N))^(1/(D+4));

        f = 1/N * sum(Kg(xq,x,h),2); %sum over the columns

        function k = Kg(x,mu,sigma)
            k = 1./(sqrt(2*pi)*sigma) .* exp(-(x-mu).^2./(2*sigma.^2));
        end
    end
end
