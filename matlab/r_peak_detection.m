
%% ECG Arrhythmia Detection & Analysis
% Phase 3A: QRS Detection - Signal Transformation
% MIT-BIH Arrhythmia Database - Record 208

clear;
clc;
close all;

%% 1. Project paths

scriptPath = mfilename('fullpath');
projectRoot = fileparts(fileparts(scriptPath));

inputFile = fullfile(projectRoot, ...
    'data', 'processed', 'record_208_filtered.csv');

outputDir = fullfile(projectRoot, 'results', 'figures');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% 2. Load filtered ECG

data = readtable(inputFile);

t = data.time_s;
ecg = data.ecg_filtered_mV;

fs = 360;

fprintf('Samples: %d\n', length(ecg));
fprintf('Sampling frequency: %d Hz\n', fs);

%% 3. Derivative

% Five-point derivative approximation
% Highlights rapid ECG changes

kernel = [1 2 0 -2 -1] * (fs / 8);

ecgDerivative = conv(ecg, kernel, 'same');

%% 4. Squaring

% Makes all values positive
% Emphasizes large slopes

ecgSquared = ecgDerivative .^ 2;

%% 5. Moving Window Integration

% Integration window: 150 milliseconds

windowDuration = 0.150;

windowSamples = round(windowDuration * fs);

ecgIntegrated = movmean( ...
    ecgSquared, windowSamples);

%% 6. Visualize first 10 seconds

n = 10 * fs;

figure('Color', 'w', ...
    'Position', [100 100 1200 850]);

subplot(4,1,1);
plot(t(1:n), ecg(1:n), 'b');
title('1. Filtered ECG');
ylabel('Amplitude (mV)');
grid on;

subplot(4,1,2);
plot(t(1:n), ecgDerivative(1:n), 'm');
title('2. Derivative');
ylabel('Slope (mV/s)');
grid on;

subplot(4,1,3);
plot(t(1:n), ecgSquared(1:n), 'r');
title('3. Squared Signal');
ylabel('Squared slope');
grid on;

subplot(4,1,4);
plot(t(1:n), ecgIntegrated(1:n), 'k');
title('4. Moving Window Integration - 150 ms');
xlabel('Time (seconds)');
ylabel('Integrated energy');
grid on;

sgtitle('QRS Detection - Signal Processing Pipeline');

%% 7. Save figure

exportgraphics(gcf, ...
    fullfile(outputDir, 'qrs_detection_pipeline.png'), ...
    'Resolution', 200);

%% 8. Export signals for Python

output = table( ...
    t, ecg, ecgDerivative, ...
    ecgSquared, ecgIntegrated, ...
    'VariableNames', { ...
    'time_s', ...
    'ecg_filtered_mV', ...
    'derivative', ...
    'squared', ...
    'integrated'});

writetable(output, ...
    fullfile(projectRoot, 'data', 'processed', ...
    'record_208_qrs_pipeline.csv'));

disp('QRS signal transformation completed.');

%% 9. Initial QRS candidate detection

% Initial threshold for exploratory detection
threshold = 0.10 * max(ecgIntegrated);

% Minimum separation between candidate QRS peaks
minDistance = round(0.20 * fs);

[pks, locs] = findpeaks( ...
    ecgIntegrated, ...
    'MinPeakHeight', threshold, ...
    'MinPeakDistance', minDistance);

candidateTimes = t(locs);

fprintf('QRS candidates detected: %d\n', length(locs));

%% 10. Plot detected candidates

figure('Color', 'w', ...
    'Position', [100 100 1200 500]);

n = 10 * fs;

plot(t(1:n), ecgIntegrated(1:n), ...
    'k', 'LineWidth', 1);
hold on;

mask = locs <= n;

plot(candidateTimes(mask), pks(mask), ...
    'ro', 'MarkerSize', 7);

yline(threshold, '--b', 'Initial threshold');

title('Initial QRS Candidate Detection - Record 208');
xlabel('Time (seconds)');
ylabel('Integrated energy');
legend('Integrated signal', 'QRS candidates', ...
    'Threshold');
grid on;

%% 11. Save candidates

candidates = table(locs, candidateTimes, pks, ...
    'VariableNames', ...
    {'sample', 'time_s', 'integrated_peak'});

writetable(candidates, ...
    fullfile(projectRoot, 'data', 'processed', ...
    'record_208_qrs_candidates.csv'));

exportgraphics(gcf, ...
    fullfile(outputDir, 'qrs_candidates.png'), ...
    'Resolution', 200);
