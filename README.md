**Purpose** Train a DNN-based meta-model to predict a primary model accuracy drop (on various shifted datasets) and beat the baseline.

**Primary Task** RandomForestClassifier to predict low/high sales of video games records. Accuracy on clean validation set 0.798.

**Data for the Performance Drop Regressor**
- training: 500 datasets (X1), their accuracy drop (y), their meta-features (X2)
- validation: take a random split of the previous, if needed.
- test data:
   1. test: 500 datasets (X1) with same shifts as in the training, but different severity (and their X2 and y).
   2. test_unseen: 900 datasets (X1) with other types of shifts, not seen at training time (and their X2 and y).
   3. test_natural: 10 datasets (X1) coming from different domains, but same primary task (and their X2 and y).
   
Each dataset has 475 rows and 9 features (preprocessed already).

Each meta-feature vector contains 114 features (will be preprocessed in this notebook to 110 final features).

**Baseline**

Baseline-Meta-Features: RandomForestRegressor trained on meta features only (prediction_percentiles, PAD, RCA, confidence drop, 
BBSDs KS and BBSDh X2 statistics, KS statistics on individual preprocessed features.

**DNN Models**

The implemented DNNs in keras use one of the 3 types of encoders 'mlp', 'lstm' or 'odt' (oblivious decision tree). 


