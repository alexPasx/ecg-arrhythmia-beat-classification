
%% ECG Arrhythmia Detection & Analysis
% Phase 2: Signal Preprocessing
% Dataset: MIT-BIH Arrhythmia Database - Record 208

clear;
clc;
close all;

%% 1. Project paths

scriptPath = mfilename('fullpath');
projectRoot = fileparts(fileparts(scriptPath));

dataFile = fullfile(projectRoot, ...
    'data', 'processed', 'record_208_5min.csv');

outputDir = fullfile(projectRoot, 'results', 'figures');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% 2. Load ECG data

data = readtable(dataFile);

t = data.time_s;
ecg = data.ecg_mV;

fs = 360;

fprintf('ECG samples: %d\n', length(ecg));
fprintf('Sampling frequency: %d Hz\n', fs);
fprintf('Duration: %.2f minutes\n', length(ecg)/fs/60);

%% 3. Butterworth band-pass filter

lowCutoff = 5;
highCutoff = 18;
filterOrder = 3;

[b, a] = butter(filterOrder, ...
    [lowCutoff highCutoff]/(fs/2), 'bandpass');

ecgFiltered = filtfilt(b, a, ecg);

%% 4. Select first 10 seconds

nSamples = 10 * fs;

t10 = t(1:nSamples);
raw10 = ecg(1:nSamples);
filtered10 = ecgFiltered(1:nSamples);

%% 5. Visualization

figure('Color', 'w', 'Position', [100 100 1200 650]);

subplot(2,1,1);

plot(t10, raw10, 'b', 'LineWidth', 1);

title('Original ECG - MIT-BIH Record 208');
xlabel('Time (seconds)');
ylabel('Amplitude (mV)');
grid on;

subplot(2,1,2);

plot(t10, filtered10, 'r', 'LineWidth', 1);

title('Filtered ECG - Butterworth 5-18 Hz');
xlabel('Time (seconds)');
ylabel('Amplitude (mV)');
grid on;

sgtitle('ECG Signal Preprocessing');

%% 6. Save figure

exportgraphics(gcf, ...
    fullfile(outputDir, 'ecg_preprocessing_comparison.png'), ...
    'Resolution', 200);

%% 7. Export filtered ECG for Python

t = t(:);
ecg = ecg(:);
ecgFiltered = ecgFiltered(:);

fprintf('Time size: %s\n', mat2str(size(t)));
fprintf('ECG size: %s\n', mat2str(size(ecg)));
fprintf('Filtered size: %s\n', mat2str(size(ecgFiltered)));

filteredData = table(t, ecg, ecgFiltered, ...
    'VariableNames', ...
    {'time_s', 'ecg_raw_mV', 'ecg_filtered_mV'});

outputFile = fullfile(projectRoot, ...
    'data', 'processed', 'record_208_filtered.csv');

writetable(filteredData, outputFile);

disp('Preprocessing completed successfully.');