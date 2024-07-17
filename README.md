# "Selective" Mechanism for Identifying Individual Unnecessary Blood Tests
Huang, T., Li, L. T., Bernstam, E. V., & Jiang, X. (2023). **Confidence-based laboratory test reduction recommendation algorithm**. *BMC Medical Informatics and Decision Making*, 23(1), 93. [Paper link](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-023-02187-3#citeas)

### Abstract
We collected internal patient data from a teaching hospital in Houston and external patient data from the MIMIC III database. The study used a conservative definition of unnecessary laboratory tests, which was defined as stable (i.e., stability) and below the lower normal bound (i.e., normality). Considering that machine learning models may yield less reliable results when trained on noisy inputs containing low-quality information, we estimated prediction confidence to assess the reliability of predicted outcomes. We adopted a “select and predict” design philosophy to maximize prediction performance by selectively considering tests with high prediction confidence for recommendations. Our model accommodated irregularly sampled observational data to make full use of variable correlations (i.e., with other laboratory test values) and temporal dependencies (i.e., previous laboratory tests performed within the same patient) in selecting tests for training and prediction.

### Introduction of "Selective" Mechanism
The study adopted the “selective” mechanism to quantify the
predictability of future lab tests. This mechanism differentiates between tests that provide
critical insights and those that are redundant. Unnecessary tests, which often replicate
information from previous tests, contribute little value to monitoring patient
condition changes due to their predictable results.

The concept of “predictability” under the “selective” mechanism is akin to
students selecting exam topics based on their difficulty. Consider two students preparing
for an exam with five distinct and unrelated topics, where they must choose only two
topics to answer. Student A studies all topics indiscriminately, whereas student B
strategically focuses on the easier topics. Analogously, student B will likely perform
better by concentrating on topics with a higher chance of success. Similarly, my study
employed selective predictions to focus on lab tests that can be predicted with high
assurance, thereby enhancing overall prediction accuracy.

The “selective” mechanism employs a selector that excludes tests that the model
“disagrees to predict” (i.e., unpredictable tests) due to the underlying difficulty in
estimating their results, focusing instead on those that the model “agrees to predict” (i.e.,
predictable tests) as the model can predict their results with high assurance. This
mechanism identifies predictable lab tests that are likely redundant, as their results can be
easily estimated by the model. For example, future lab tests are predictable because their
past observations do not significantly change over time, and there is no noise in the
records. The operation of this model involves a user-defined threshold that determines the
categorization of predictable versus unpredictable tests.

Practically, the model’s ability to select lab tests not only optimizes the testing
process but also enhances patient safety. By obtaining high accuracy in the tests the
model selects, the model can minimize the risk of misclassification and reduce potential
errors in patient monitoring.

### Dataset
Due to privacy policy, we only provide data collection and pre-processing approaches for MIMIC-III. First, download data into the `mimic_data` folder. The dataset is publicly available on PhysioNet: https://physionet.org/content/mimiciii/1.4/

