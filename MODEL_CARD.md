# Model card — BrainGuard tumour-type classifier

Format: Mitchell et al., *Model Cards for Model Reporting* (2019). Performance figures are generated into `reports/results.md` by the GPU run and copied here from that file only.

## Model details
- **Type:** ResNet50 (ImageNet initialisation), fine-tuned to classify one contrast-enhanced T1 MRI slice as glioma, meningioma or pituitary tumour. Selected from six architectures compared on the same patient folds (EfficientNetB0, the pre-specified model, was 3.7 points lower; see the README).
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
| Balanced accuracy | 90.1% (87.2–92.5%) |
| Accuracy | 91.8% (89.3–93.9%) |
| Macro AUC | 0.983 (0.972–0.991) |
| ECE / Brier | 0.036 / 0.126 (≥ 99% confidence: 98.5% correct; over-confident below 90%, so re-calibrate before showing probabilities) |
| Sensitivity: glioma / meningioma / pituitary | 95.0% / **78.2%** / 97.2% |
| Specificity: glioma / meningioma / pituitary | 94.7% / 96.6% / 96.0% |

A second training run on the same folds gave 90.7% balanced accuracy and 85.2% meningioma sensitivity: overall accuracy is stable between runs, meningioma sensitivity less so. The model was selected on these folds, so the estimate is slightly optimistic.

**Robustness** (balanced-accuracy drop on held-out patients):
- Gaussian noise: −6.0 (sd 8) to −10.6 points (sd 16);
- blur σ 1.5: −9.2;
- half resolution: −4.2;
- brightness, contrast, gamma, 10° rotation and JPEG 30: within 1.1.

## Explanations
SHAP values (expected gradients) and Grad-CAM are produced for 150 held-out images and tested against the radiologists' tumour masks:
- Both put more attribution on the tumour than chance (about 3.6×), but most attribution still lies outside it (median 4.6% SHAP, 4.5% Grad-CAM inside a tumour that covers 1.2% of the image).
- **Grad-CAM passes** the randomisation sanity check (similarity to a random model −0.06).
- **SHAP is borderline** (0.53): its maps partly reflect image structure. Show Grad-CAM, not SHAP, to a reader, and never present either as a tumour outline.

## Known limitations and risks
- **Single source:** one public dataset from two hospitals. There is no external test set from another institution, so generalisation to other scanners is unproven. The robustness tests only simulate it.
- **2-D slices:** the model sees one slice and no patient-level context.
- **Meningioma:** the smallest class (23%) and the weakest (78–85% sensitivity across two training runs). The model must not be used to rule meningioma out.
- **Image quality:** noise, blur and low resolution reduce balanced accuracy by 4–11 points; this model is more sensitive to noise than EfficientNetB0.
- **Benchmark contamination:** the public Kaggle benchmark that most portfolio models use shares scans and patients between its splits, and its no-tumour class is confounded with image source (see `reports/kaggle_*.md`). Accuracy figures from such benchmarks are not comparable with these results.

## Ethical considerations
The data are de-identified public research data. A confident wrong tumour type could mislead a non-expert, so outputs must never be shown without the explanation maps and the limitations above.
