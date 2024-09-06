import numpy as np
import pandas as pd

sub1 = pd.read_csv("submission/clf_submission_3600_50_ctgan.csv") # LB 0.8098937489 
sub2 = pd.read_csv("submission/clf_submission_3600_50.csv") # CV 74.47 LB 0.8098533184
sub3 = pd.read_csv("submission/clf_submission_2700_10.csv") # CV 74.7x LB 0.809304323 
sub4 = pd.read_csv("submission/clf_submission_3600_50_40_ctgan.csv") # CV 74.91
sub5 = pd.read_csv("submission/clf_submission_3000_30_40_ctgan.csv") # CV 75.19

preds = pd.DataFrame({
    'sub1': sub1.iloc[:, 1],
    'sub2': sub2.iloc[:, 1],
    'sub3': sub3.iloc[:, 1],
    'sub4': sub4.iloc[:, 1],
    'sub5': sub5.iloc[:, 1]
})

def hard_ensemble(row):
    weak_labels = ['d', 'g', 'c', 'h', 'j']
    for label in weak_labels:
        if (row['sub1'] == 'm' and row['sub2'] == 'm' and row['sub3'] == 'm' and row['sub5'] == label):
            return label
        
    weak_labels = ['d', 'g', 'c', 'j', 'i']
    for label in weak_labels:
        if (row['sub1'] == 'm' and row['sub2'] == 'm' and row['sub3'] == 'm' and row['sub4'] == label):
            return label
        
    weak_labels = ['d', 'g', 'c', 'j', 'i', 'h']
    for label in weak_labels:
        if (row['sub1'] == 'm' and row['sub2'] == 'm' and row['sub3'] == label) or \
        (row['sub1'] == 'm' and row['sub2'] == label and row['sub3'] == 'm') or \
        (row['sub1'] == label and row['sub2'] == 'm' and row['sub3'] == 'm'):
            return label
        
    if (row['sub1'] == 'g' and row['sub2'] == 'g' and row['sub5'] == 'j') or \
        (row['sub1'] == 'j' and row['sub2'] == 'j' and row['sub5'] == 'g'):
            return row['sub5']
    
    if row['sub1'] == row['sub2'] == row['sub3']:  # 모두 동일한 경우
        return row['sub1']
    elif row['sub1'] == row['sub2']:  # sub1과 sub2가 동일한 경우
        return row['sub1']
    elif row['sub1'] == row['sub3']:  # sub1과 sub3가 동일한 경우
        return row['sub1']
    elif row['sub2'] == row['sub3']:  # sub2와 sub3가 동일한 경우
        return row['sub2']
    else:  # 모두 다르다면 sub1 우선
        return row['sub1']
    
preds['hard'] = preds.apply(hard_ensemble, axis=1)
print(preds['hard'].value_counts())

submission = pd.read_csv("data/sample_submission.csv")
submission['Fraud_Type'] = preds['hard']
submission.to_csv('submission/clf_submission.csv', index=False)