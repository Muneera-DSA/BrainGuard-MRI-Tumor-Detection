# Model card — BrainGuard tumour-type classifier

Format: Mitchell et al., *Model Cards for Model Reporting* (2019). Performance figures are generated into `reports/results.md` by the GPU run and copied here from that file only.

## Model details
- **Type:** EfficientNetB0 (ImageNet initialisation), fine-tuned to classify one contrast-enhanced T1 MRI slice as glioma, meningioma or pituitary tumour.
- **Training data:** figshare brain tumour dataset (Cheng et al., CC BY 4.0): 3,064 slices from 233 patients, two hospitals in China, 2005–2010.
- **Validation:** 5-fold cross-validation grouped by patient; early stopping on separate training patients.
- **Author:** Muneera Mohamed. Portfolio research project; not deployed.

## Intended use
- Research and education: demonstrating leakage-safe evaluation and explanation testing for medical imaging models.
- At most, a second-reader aid that suggests a tumour **type** for a slice already known to contain a tumour, with a clinician deciding.

## Out-of-scope uses
- **Tumour detection** (tumour vs no tumour). The model has never seen a scan without a tumour, and it will assign one of three tumour types to any input.
- Any diagnostic, triage or treatment decision without a qualified clinician.
- Other MRI sequences (T2, FLAIR, non-contrast T1), other organs, paediatric scans, or 3-D volumes.
- Populations, scanners or protocols not represented in the training data, without re-validation.

## Performance
From `reports/results.md`. Evaluation: 5-fold CV grouped by patient, 3,064 slices from 233 patients. 95% CIs from a patient-level bootstrap.

| Metric | Value (95% CI) |
|---|---|
| Balanced accuracy | 86.5% (83.5–89.1%) |
| Accuracy | 88.4% (85.5–91.1%) |
| Macro AUC | 0.971 (0.951–0.985) |
| ECE / Brier | 0.045 / 0.170 (over-confident between 60% and 95%; re-calibrate before showing probabilities) |
| Sensitivity: glioma / meningioma / pituitary | 91.2% / **69.9%** / 98.3% |
| Specificity: glioma / meningioma / pituitary | 92.7% / 95.2% / 94.3% |

**Robustness** (balanced-accuracy drop on held-out patients):
- blur σ 1.5: −11.3 points;
- Gaussian noise: −5.7 to −6.5;
- half resolution: −4.6;
- contrast ×0.7: −4.0;
- brightness, gamma, 10° rotation and JPEG 30: within ±1.7.

## Explanations
SHAP values (expected gradients) and Grad-CAM are produced for 150 held-out images and tested against the radiologists' tumour masks:
- Both put more attribution on the tumour than chance (about 2–2.5×). Most attribution still lies outside it (median 3.5% SHAP, 2.5% Grad-CAM inside).
- **Grad-CAM passes** the randomisation sanity check (similarity to a random model 0.16).
- **SHAP does not** (0.77): its maps mostly reflect image structure. Show Grad-CAM, not SHAP, to a reader, and never present either as a tumour outline.

## Known limitations and risks
- **Single source:** one public dataset from two hospitals. There is no external test set from another institution, so generalisation to other scanners is unproven. The robustness tests only simulate it.
- **2-D slices:** the model sees one slice and no patient-level context.
- **Meningioma:** the smallest class (23%) and the weakest (69.9% sensitivity). The model must not be used to rule meningioma out.
- **Image quality:** blur, noise and low resolution reduce balanced accuracy by 5–11 points.
- **Benchmark contamination:** the public Kaggle benchmark that most portfolio models use shares scans and patients between its splits, and its no-tumour class is confounded with image source (see `reports/kaggle_*.md`). Accuracy figures from such benchmarks are not comparable with these results.

## Ethical considerations
The data are de-identified public research data. A confident wrong tumour type could mislead a non-expert, so outputs must never be shown without the explanation maps and the limitations above.
