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
Across analyses, the results provide evidence for the presence of **individualized attention patterns**, though these appear primarily at the session level rather than as stable, long-term traits. The per-session regression analysis showed that participants’ gaze entropy trajectories only partially aligned with the group mean, with an average R² of about 0.21. This indicates that while the stimulus exerts a strong influence on visual behavior, participants also display unique temporal patterns of attention. The wide range of slopes (−0.4 to 2.1) and intercepts (−3.3 to 4.1) further suggests meaningful diversity in focus style—some individuals consistently maintained tighter, more concentrated gaze behavior, whereas others adopted a more exploratory viewing approach. Although the mixed-effects model revealed limited between-participant variance (≈ 0.005), the consistent deviations from the group trajectory across sessions imply that personalization in attention does exist. Participants express distinct, session-specific gaze dynamics shaped by their individual cognitive or perceptual tendencies, even when viewing identical stimuli.



<img width="984" height="785" alt="image" src="https://github.com/user-attachments/assets/3088bc5d-b68f-4f93-9819-7e9e3c4da18b" />

