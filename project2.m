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
% subplot(2,1,1);
autocorr(log_rtns_m);
saveas(gcf,'plots/acf_log_rtns.png');
% subplot(2,1,2);

% Ljung-box test
[~, pValue] = lbqtest(log_rtns_m, 'lags', 20);
fprintf("The p-value from the Ljung-box test is: %.04f\n", pValue);
% Fail to reject null hypothesis that log returns come from IIDN
figure;
parcorr(log_rtns_m);
saveas(gcf,'plots/pacf_log_rtns.png');

% Plot the autocorrelation of the squared returns
figure;
autocorr(log_rtns_m.^2);
title("Sample Squared Returns Autocorrelation");
saveas(gcf, 'plots/acf_square_rtns.png');

%% Problem 2
n = length(log_rtns_m);
training = log_rtns_m(1:1000);
test = log_rtns_m(1001:end);

normBIC = zeros(10, 10);

for p = 1:10
    for q = 1:10
        % Create the GARCH(p,q) model without an offset
        mdl = garch('GARCHLags', p,'ARCHLags',q);
        % Estimate the model for the training data
        [estMdl, ~, ~] = estimate(mdl, training, 'Display', 'off');        
        results = estMdl.summarize;
        normBIC(p,q) = results.BIC;
    end
end

% Plot heatmap of BIC for each of the GARCH(p,q) models
figure;
colormap('jet');
imagesc(normBIC);
set(gca,'YDir','normal') ;
colorbar;
title('BIC for GARCH(p,q) models');
xlabel('P');
ylabel('Q');
saveas(gcf,'plots/bic_heatmap_norm.png');

% The minimum value is 1, which corresponds to GARCH(1,1)
[min_bic, min_bic_idx] = min(normBIC, [], 'all', 'linear');
% Get the min p and q
[min_p, min_q] = ind2sub([10,10], min_bic_idx);
fprintf("Minimum BIC for normal distribution model: %.03f\n", min_bic);

%% Problem 3 
% GARCH(1,1)
norm_mdl = garch(1,1);
[norm_est_mdl, ~, norm_logL] = estimate(norm_mdl, training);

v = infer(norm_est_mdl, training);

% Compute the residuals
res = training ./ sqrt(v);

figure;
subplot(2,2,1);
plot(res);
title('Standardized Residuals');
subplot(2,2,2);
qqplot(res);
title({'QQ Plot of Residuals vs','Standard Normal Distribution'});
subplot(2,2,3);
autocorr(res);
title('Residual Sample ACF');
subplot(2,2,4);
autocorr(res.^2);
title('Residual^2 Sample ACF');
saveas(gcf,'plots/residual_plots_norm.png');

% If the model is correct the distribution of the residual should be normal
% What about the covariance structure (is this why we do Residual^2?
% What does the autocorrelation plot of the residuals look like?
% What does the autocorrelation plot of the squared residuals look like?
% What about a qq-plot of the residuals? Are there problems?

%% Problem 4
% Repeat problem 2 & 3 for GARCH(p,q) with Student's t distribution

tBIC = zeros(10, 10);
for p = 1:10
    for q = 1:10
        % Create the GARCH(p,q) model without an offset
        mdl = garch('GARCHLags', p,'ARCHLags',q,'Distribution','t');
        % Estimate the model for the training data
        [estMdl, ~, ~] = estimate(mdl, training, 'Display', 'off');        
        results = estMdl.summarize;
        tBIC(p,q) = results.BIC;
    end
end

% Plot heatmap of BIC for each of the GARCH(p,q) models with
%  student's t distribution
figure;
colormap('jet');
imagesc(tBIC);
set(gca,'YDir','normal') ;
colorbar;
title('BIC for GARCH(p,q) models');
xlabel('P');
ylabel('Q');
saveas(gcf,'plots/bic_heatmap_t.png');

% Get the min index, and convert back to (p,q)
[min_bic_t, min_bic_idx_t] = min(tBIC, [], 'all', 'linear');
[t_min_p, t_min_q] = ind2sub([10,10], min_bic_idx_t);
fprintf("Minimum BIC for t-distribution model: %.03f\n", min_bic_t);

%% Problem 4
% GARCH(1,1)
tMdl = garch('GARCHLags', t_min_p,'ARCHLags', t_min_q, 'Distribution', 't');
tEstMdl = estimate(tMdl, training); 

v = infer(tEstMdl, training);
% Compute the residuals
res = training ./ sqrt(v);

figure;
subplot(2,2,1);
plot(res);
title('Standardized Residuals');
subplot(2,2,2);
% QQ-plot for t distribution with the estimated degrees of freedom
tdist = makedist('tLocationScale', 'nu', tEstMdl.Distribution.DoF);
qqplot(res, tdist);
title({'QQ Plot of Residuals vs',"Student's t Distribution", "(df = " + round(tEstMdl.Distribution.DoF, 2) + ")"});
xlabel("Quantiles of t distribution");
subplot(2,2,3);
autocorr(res);
title('Residual Sample ACF');
subplot(2,2,4);
autocorr(res.^2);
title('Residual^2 Sample ACF');
saveas(gcf,'plots/residual_plots_t.png');



%% Problem 5
% Compute the inverse cdf values for norm and t-distribution using
% estimated degrees of freedom from the model.
alpha = 0.05;
z_05 = norminv(1 - alpha / 2);
t_05 = tinv(1 - alpha  / 2, tEstMdl.Distribution.DoF);


norm_forecasts = zeros(length(test),1);
t_forecasts = zeros(length(test), 1);
simple_forecasts = zeros(length(test), 1);

for i = 1:length(test)    
    % Use all of the data up until the prediction for each forecast    
    norm_forecasts(i) = forecast(norm_est_mdl, 1, log_rtns_m(1:(1000 + i)));
    t_forecasts(i) = forecast(tEstMdl, 1, log_rtns_m(1:(1000 + i)));
    simple_forecasts(i) = var(log_rtns_m(1:(1000 + i)));
end

norm_ci = z_05 .* sqrt(norm_forecasts);
t_ci = t_05 .* sqrt(t_forecasts);
simple_ci = z_05 .* sqrt(simple_forecasts);

% Plot CIs as lines
% Overlay all the CIs on the same plot
test_x = 1000 + (1:length(test));
xl = [min(test_x), max(test_x)];
figure;
p_d = plot(test_x, test, 'Color', [.7,.7,.7]);
xlim(xl);
hold on;
p_n = plot(test_x, norm_ci, 'r');
plot(test_x, - norm_ci, 'r');
p_t = plot(test_x, t_ci, 'b');
plot(test_x, - t_ci, 'b');
p_s = plot(test_x, simple_ci, 'black:', 'LineWidth',2);
plot(test_x, - simple_ci, 'black:', 'LineWidth',2);
legend([p_d, p_n, p_t, p_s] , 'Data', 'Normal', 'T', 'Simple');
title("Confidence intervals for the three models");
saveas(gcf,'plots/conf_ints_overlay.png');

% Count the number of times the log returns fall into the CIs
norm_ci_count = sum(test > -norm_ci & test < norm_ci);
t_ci_count = sum(test > -t_ci & test < t_ci);
simple_ci_count = sum(test > -simple_ci & test < simple_ci);

counts = [norm_ci_count, t_ci_count, simple_ci_count];
counts(2,:) = round(counts(1,:) / 251, 3);
% Pretty print the table of counts and percentages
array2table(counts, 'RowNames', {'Count', '%'} ,'VariableNames', {'Normal', 'T', 'Simple'})

%% Testing other models
% GJR(1,1) performs better according to BIC
% https://se.mathworks.com/help/econ/specify-gjr-models-using-gjr.html
% https://se.mathworks.com/help/econ/compare-garch-and-egarch-fits.html
% Can do a likelihood ratio test: The GARCH(1,1) is nested in the GJR(1,1) model, however, so you could use a likelihood ratio test to compare these models.
% Found that having error with normal distribution performs better than
% t-distribution for GJR model
gjr_mdl = gjr(1,1);
[gjr_mdl_est, gjr_estParam, gjr_logL] = estimate(gjr_mdl, training);
gjr_results = gjr_mdl_est.summarize;
gjr_bic = gjr_results.BIC;
% Get the min p and q
fprintf("BIC for GJR model: %.03f\n", gjr_bic);

% Can compare the normal GARCH model and GJR model because they are nested
% Norm model

% Likelihood ratio test
% Reject the null hypothesis, and conclude that we need the 
[h, p] = lratiotest(gjr_logL, norm_logL, 1);