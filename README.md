# ECG Arrhythmia Detection & Analysis

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB)

Offline classification of annotated ECG beats from the MIT-BIH Arrhythmia Database. Four classical models predict one label per beat: **N** (normal), **V** (premature ventricular contraction), or **F** (fusion of a ventricular and a normal beat). Both evaluation protocols split by patient group and score individual beats. The repository also contains exploratory QRS detection scripts in Python and MATLAB.

## Repository structure

```text
.
├── python/                 # Data download, feature extraction, training and evaluation
├── matlab/                 # Exploratory ECG preprocessing and QRS detection
├── results/                # Evaluation metrics and confusion matrices
│   └── group_holdout/     # Holdout metrics and confusion matrices
├── requirements.txt       # Pinned Python dependencies
└── README.md
```

## Data and features

The project uses the original WFDB symbols N, V and F from the reference `atr` annotations. Other annotation symbols are excluded as targets. This is a three-symbol task, distinct from the five-class AAMI mapping described by de Chazal et al.

The database contains 48 recordings from 47 subjects. The feature table contains **82,953 beats from 47 recordings and 46 patient groups**. Record 232 contributes no N, V or F beats. Records 201 and 202 are assigned to the same group because they come from the same subject.

| Beat symbol | Retained beats |
| --- | ---: |
| N | 75,022 |
| V | 7,129 |
| F | 802 |

The extractor uses the first recorded channel at 360 Hz and applies a third-order Butterworth 5–18 Hz bandpass in forward and reverse directions to each recording. It extracts a window from 250 ms before to 450 ms after the reference annotation, skipping windows that cross a recording boundary. Each beat has 260 features: 252 filtered samples, previous and next RR intervals, and six amplitude, energy and slope statistics. Each RR interval is the time to the nearest earlier or later reference annotation in the same recording whose symbol is in `BEAT_SYMBOLS`: `N L R B A a J S V r F e j n E / f Q ?`. This includes paced beats (`/`) and skips annotations outside that list, such as the signal-quality marker (`~`).

## Evaluation and results

Both protocols keep patients disjoint across training and evaluation. Record ID, sample index and target label are excluded from the feature matrix. Median imputation and, for KNN and both SVMs, standardization are fitted inside each training pipeline. Random Forest uses median imputation without scaling. KNN uses five neighbours. Both SVMs use C = 1. The SVMs and the 200-tree Random Forest use balanced class weights.

Macro F1 gives equal weight to the three beat classes. Weighted F1 weights each class by its beat count. In cross-validation, pooled metrics combine all out-of-fold predictions before scoring. Fold mean and sample SD are computed across the five fold scores. Neither aggregation averages scores by patient.

### Five-fold patient-grouped cross-validation

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` assigns each patient group to one evaluation fold. Fold 1 through 5 contain 13, 7, 8, 10 and 8 patient groups, with 30, 367, 8, 3 and 394 F beats, respectively.

| Model | Pooled accuracy | Pooled macro F1 | Pooled weighted F1 | Fold macro F1 mean ± SD | Pooled F precision | Pooled F recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KNN | 0.9328 | 0.5687 | 0.9353 | 0.5696 ± 0.0350 | 0.45% | 0.50% |
| Linear SVM | 0.9502 | **0.6565** | 0.9523 | 0.6332 ± 0.0783 | 17.08% | 20.45% |
| RBF SVM | 0.8642 | 0.5789 | 0.9034 | 0.5848 ± 0.0428 | 0.58% | 5.49% |
| Random Forest | 0.9412 | 0.5716 | 0.9405 | 0.5761 ± 0.0451 | 0.00% | 0.00% |

Source files: [pooled metrics](results/ml_5fold_pooled.csv), [fold metrics](results/ml_5fold_results.csv) and [fold summary](results/ml_5fold_summary.csv). The confusion matrices in `results/` use true labels as rows and predicted labels as columns.

### 75/25 patient-grouped holdout

One `GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)` split. The 75/25 target applies to patient groups. The actual split has 34 training groups (58,018 beats) and 12 test groups (24,935 beats). Test support is 22,282 N, 2,625 V and 28 F beats.

| Model | Accuracy | Macro F1 | Weighted F1 | F precision | F recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| KNN | 0.9265 | **0.5675** | 0.9329 | 1.26% | 7.14% |
| Linear SVM | 0.8766 | 0.5311 | 0.8981 | 0.52% | 10.71% |
| RBF SVM | 0.7772 | 0.4854 | 0.8302 | 0.19% | 14.29% |
| Random Forest | 0.8634 | 0.5262 | 0.8894 | 0.14% | 3.57% |

The [holdout results](results/group_holdout/holdout_results.csv) include precision, recall and F1 for N, V and F. The test set comes from the same source cohort as the cross-validation experiment.

## Limitations

- **Class imbalance and limited F support.** N accounts for about 90.4% of the extracted beats. Across the four holdout models, only one to four of the 28 F beats are correctly classified. Low F precision makes overall accuracy a poor guide to performance on this class.
- **Sensitivity to the split.** Linear SVM leads on pooled cross-validation macro F1, while KNN leads on the single holdout. The protocols use different training sizes and class distributions. Their scores provide internal comparisons on one database. No external validation cohort was evaluated.
- **Reference annotations and future context.** Beat positions and RR intervals use reference annotations. The post-beat window, next RR interval and forward–reverse filtering use future signal or annotation information. The results describe offline classification given reference beat locations. The separate QRS experiments do not establish end-to-end classification performance.
- **Benchmark comparability.** The three-symbol task focuses on distinguishing normal, PVC and fusion beats with their original WFDB labels. Scores are specific to that label set and these splits, so they cannot be compared directly with five-class AAMI results.
- **Scope.** Evaluation covers the first recorded channel and three selected annotation symbols. The reported metrics do not establish patient-level diagnostic or clinical performance.

## Reproduce the classification experiments

From the repository root, use Python 3.11 (local audit used 3.11.9). The following commands are for PowerShell. The downloader and feature extractor query PhysioNet for the record list. The run needs local disk space for the raw signals and a feature CSV of about 446 MB.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe python/download_all_records.py
.\.venv\Scripts\python.exe python/extract_ml_features.py
.\.venv\Scripts\python.exe python/check_ml_dataset.py
.\.venv\Scripts\python.exe python/create_cv_folds.py
.\.venv\Scripts\python.exe python/train_5fold_cv.py
.\.venv\Scripts\python.exe python/train_group_holdout.py
```

The training commands regenerate CSV reports in `results/` and `results/group_holdout/` and overwrite prior reports at those paths. Raw records, features, per-beat predictions, split exports and figures are local artifacts. `train_ml_classifiers.py` is an exploratory fold-1 baseline. The exploratory MATLAB preprocessing and QRS scripts use Signal Processing Toolbox and record 208 exports. The empty test and model scaffolds do not provide automated coverage or implemented neural models.

The archived runs logged about **13 min** of model fitting across all models and CV folds and **1 min 49 s** for the holdout. These totals exclude downloads, feature extraction and prediction. The CPU model was not recorded with the run, and end-to-end runtime depends on the machine.

The archived reports were checked by regenerating both splits and recalculating scores from saved predictions in the pinned local environment. Model training and the MATLAB scripts were not rerun during that audit.

## Data attribution

The results contain information from the [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/) by George B. Moody and Roger G. Mark, provided by PhysioNet under the [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/). The recordings and annotations are downloaded separately from PhysioNet.

## References

1. Moody, G. B., and Mark, R. G. [MIT-BIH Arrhythmia Database, version 1.0.0](https://physionet.org/content/mitdb/1.0.0/). PhysioNet. DOI 10.13026/C2F305. See also the [database directory](https://physionet.org/content/mitdb/1.0.0/mitdbdir/src/intro.tr) for recording details.
2. PhysioNet. [WFDB annotation codes](https://physionet.org/physiotools/wpg/wpg_36.htm).
3. de Chazal, P., O'Dwyer, M., and Reilly, R. B. (2004). [Automatic classification of heartbeats using ECG morphology and heartbeat interval features](https://pubmed.ncbi.nlm.nih.gov/15248536/). IEEE Transactions on Biomedical Engineering, 51(7), 1196–1206. DOI 10.1109/TBME.2004.827359.
4. scikit-learn. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) and [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
