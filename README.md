# GazeBaseVR Attention Analysis

This repository provides a reproducible workflow for analyzing and visualizing attention patterns from the **GazeBaseVR** eye-tracking dataset.  
It combines signal preprocessing, entropy-based attention modeling, and statistical testing to explore whether gaze dynamics reveal stable, personalized attention signatures across sessions.

---

## Dataset

The **GazeBaseVR** dataset was published as:

> Lohr, D., Aziz, S., Friedman, L. et al.  
> *GazeBaseVR, a large-scale, longitudinal, binocular eye-tracking dataset collected in virtual reality.*  
> *Scientific Data*, **10**, 177 (2023).  
> [https://doi.org/10.1038/s41597-023-02075-5](https://doi.org/10.1038/s41597-023-02075-5)

Download the dataset from Figshare:  
[https://figshare.com/articles/dataset/GazeBaseVR_Data_Repository/21308391?file=38844024](https://figshare.com/articles/dataset/GazeBaseVR_Data_Repository/21308391?file=38844024)

After downloading, extract the CSV files into the following path relative to your notebook:
<pre>
data/
├── S_1002_S1_3_VID.csv
├── S_1002_S2_3_VID.csv
├── S_1009_S1_3_VID.csv
└── ...
</pre>
The analysis scripts and notebook will automatically read from the `data/` directory.

---
<pre>
gaze_analysis/
│
├── attention_metrics.py # Entropy, fixation, and dispersion metrics
├── batch_analysis.py # Batch processing for multiple participants
├── io_utils.py # Data I/O, normalization, and metadata extraction
├── preprocess.py # Cleaning, interpolation, and velocity computation
├── stat_analysis.py # Statistical tests (correlation, ANOVA, regression)
├── velocity_analysis.py # Velocity-based saccade/fixation segmentation
├── visualization.py # Visualization utilities for entropy and user/group comparisons
└── init.py
│
├── processed/ # Processed gaze data - generated after running the notebook (per participant/session)
└── analyze.ipynb # Example Jupyter notebook for running the full analysis
</pre>

---

## Workflow Overview

1. **Preprocessing**
   - Cleans and interpolates missing gaze samples.
   - Computes gaze velocity, adaptive saccade thresholds, and rolling entropy.
   - Normalizes timestamps and adds session/participant metadata.

2. **Attention and Entropy Metrics**
   - Rolling gaze entropy as a measure of focus stability.
   - Fixation-based dispersion and attention entropy.
   - Temporal entropy drift for within-session attention tracking.

3. **Statistical Analysis**
   - *Intra-subject correlation*: tests consistency of attention entropy across sessions.
   - *Mixed-effects ANOVA*: quantifies participant-level variance.
   - *User vs. Group regression*: measures how much each participant’s gaze behavior deviates from the population mean.

4. **Visualization**
   - Gaze trajectory and velocity plots.
   - Temporal attention drift (user vs. group and per-session).
   - Heatmaps of entropy over normalized video time.
   - Per-user uniqueness and focus tendency summaries.

---

## Research Question and Findings

**Research Question**:
Do participants in the GazeBaseVR dataset exhibit stable, personalized attention patterns across repeated VR video sessions?

**Findings**:
Results indicate that while individuals differ modestly in their mean gaze entropy, these differences are not stable across sessions.
Intra-subject correlations between sessions were near zero, suggesting weak personal consistency.
Mixed-effects modeling revealed low participant-level variance (≈0.005), indicating that most variance arises within individuals rather than between them.
Regression analysis showed low alignment with group patterns (mean R² ≈ 0.21), highlighting session-specific but not persistent individuality.

**Conclusion**:
Attention behavior in GazeBaseVR appears primarily stimulus-driven rather than reflecting stable personal traits.
Participants demonstrate distinct gaze entropy profiles within each session, but these do not persist across sessions, suggesting that gaze dynamics are shaped more by the visual stimulus than by enduring individual attention styles.


<img width="984" height="785" alt="image" src="https://github.com/user-attachments/assets/61ba8364-7e1c-45bc-9423-a807e85f1f19" />


