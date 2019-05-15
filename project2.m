data = importdata('speur3505ydaily.mat');
n_data = length(data);

log_data = log(data);

plot(log_data);

%% Problem 1
log_rtns = diff(log_data);
% Mean correct log returns
log_rtns_m = log_rtns - mean(log_rtns);

% Plot the autocorrelation and partial autocorrelation
figure;
subplot(2,1,1);
autocorr(log_rtns_m);
subplot(2,1,2);

% Ljung-box test
[h,pValue] = lbqtest(log_rtns_m, 'lags', 20);
% Fail to reject null hypothesis that log returns come from IIDN
parcorr(log_rtns_m);

% TODO: Do you think the returns can/should be modeled as white noise? 
% TODO: Anything that suggests that a GARCH model could be a good idea?

%% Problem 2
n = length(log_rtns_m);
training = log_rtns_m(1:1000);
test = log_rtns_m(1001:end);

logL = zeros(10,10);
numParams = zeros(10,10);

for p = 1:10
    for q = 1:10
        % Create the GARCH(p,q) model without an offset
        mdl = garch('GARCHLags', p,'ARCHLags',q);
        % Estimate the model for the training data
        [~, estParamCov1, logL(p,q)] = estimate(mdl, training, 'Display', 'off'); 
        % Save the number of non-zero parameters estimated
        numParams(p,q) = sum(any(estParamCov1));
    end
end

flatLogL = reshape(logL, [1, 100]);
flatNumParams = reshape(numParams, [1, 100]);
[aic,bic] = aicbic(flatLogL,flatNumParams,n);

% Plot heatmap of BIC for each of the GARCH(p,q) models
colormap('jet');
imagesc(reshape(bic, [10,10]));
set(gca,'YDir','normal') ;
colorbar;
title('BIC for GARCH(p,q) models');
xlabel('P');
ylabel('Q');

% The minimum value is 1, which corresponds to GARCH(1,1)
[~, min_bic_idx] = min(bic, [], 'all', 'linear');

%% Problem 3 
% GARCH(1,1)
mdl = garch('GARCHLags', p,'ARCHLags',q);
estMdl = estimate(mdl, training); 

v = infer(estMdl, training);
% Plot estimated conditional variances 
% plot(v);

res = training ./ sqrt(v);

figure;
subplot(2,2,1);
plot(res);
title('Standardized Residuals');
subplot(2,2,2);
qqplot(res);
%title('Residual Histogram');
subplot(2,2,3);
autocorr(res);
title('Residual Sample ACF');
subplot(2,2,4);
autocorr(res.^2);
title('Residual^2 Sample ACF');
%parcorr(res);
% title('Residual Sample PACF');

%% Other
% Simulate some data to see if it looks like our data
% [vS,yS] = simulate(estMdl,length(training));

% vf = forecast(estMdl, length(test), training);

% Predict 
%[V, logL] = infer(estMdl, test);

%figure;
%plot(1:1000, training);
%hold on;
%plot(1001:1251, vf, 'r');
%plot(1001:1251, sqrt(V), 'r');
%plot(1001:1251, test, ':');
