# 제출 파일 생성 관련
import os
import zipfile

# 데이터 처리 및 분석
import pandas as pd
import numpy as np
import seaborn as sns
import json
import pickle
from scipy import stats
from tqdm import tqdm

# 머신러닝 전처리
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 머신러닝 모델
import xgboost as xgb
import lightgbm as lgb
import torch

# 합성 데이터 생성
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.sampling import Condition

# To ignore all warnings
import warnings
warnings.filterwarnings('ignore')

total_df = pd.read_csv("data/train.csv")

total_df['Time_diff_last_atm_resume'] = pd.to_datetime(total_df['Transaction_resumed_date']) - pd.to_datetime(total_df['Last_atm_transaction_datetime'])
total_df['Time_diff_last_bank_branch_resume'] = pd.to_datetime(total_df['Transaction_resumed_date']) - pd.to_datetime(total_df['Last_bank_branch_transaction_datetime'])
total_df['Time_diff_transaction_resume'] = pd.to_datetime(total_df['Transaction_Datetime']) - pd.to_datetime(total_df['Transaction_resumed_date'])
total_df['Time_diff_transaction_creation'] = pd.to_datetime(total_df['Transaction_Datetime']) - pd.to_datetime(total_df['Account_creation_datetime'])
total_df['Transaction_Age'] = pd.to_datetime(total_df['Transaction_Datetime']).dt.year - total_df['Customer_Birthyear']
total_df['Transaction_age_group'] = pd.cut(total_df['Transaction_Age'], bins=[-1, 19, 29, 39, 49, 59, 99], labels=[0, 1, 2, 3, 4, 5])
total_df['Transaction_age_group'] = total_df['Transaction_age_group'].astype('int')

# datetime 변환
datetime_columns = ['Time_difference', 'Time_diff_last_atm_resume', 'Time_diff_last_bank_branch_resume', 'Time_diff_transaction_resume', 'Time_diff_transaction_creation']
for col in datetime_columns:
    total_df[col] = pd.to_timedelta(total_df[col], errors='coerce').dt.total_seconds()
    total_df[col] = total_df[col].fillna(0)
    
total_df['Customer_flag_change_of_authentications'] = total_df['Customer_flag_change_of_authentication_1'] + total_df['Customer_flag_change_of_authentication_2'] + total_df['Customer_flag_change_of_authentication_3'] + total_df['Customer_flag_change_of_authentication_4']
total_df['rooting_jailbreak_roaming_VPN'] = total_df['Customer_rooting_jailbreak_indicator'] + total_df['Customer_mobile_roaming_indicator'] + total_df['Customer_VPN_Indicator']
total_df['Customer_flag_terminal_malicious_behaviors'] = total_df['Customer_flag_terminal_malicious_behavior_1'] + total_df['Customer_flag_terminal_malicious_behavior_2'] + total_df['Customer_flag_terminal_malicious_behavior_3'] + total_df['Customer_flag_terminal_malicious_behavior_4'] + total_df['Customer_flag_terminal_malicious_behavior_5'] + total_df['Customer_flag_terminal_malicious_behavior_6']
total_df['Customer_atm_limits'] = total_df['Customer_inquery_atm_limit'] + total_df['Customer_increase_atm_limit']
total_df['Unused_suspend_status'] = total_df['Unused_account_status'] + total_df['Recipient_account_suspend_status']
total_df['Channel_OS'] = total_df['Channel'] + "_" + total_df["Operating_System"]

total_df['ChannelOS_Account_initial_balance_mean'] = total_df.groupby('Channel_OS')['Account_initial_balance'].transform('mean')
total_df['Account_account_type_Account_initial_balance_min'] = total_df.groupby('Account_account_type')['Account_initial_balance'].transform('min')

# outlier 처리
non_outliers = total_df[(total_df['Time_difference'] >= 0) & (total_df['Time_difference'] < 700000)]
mean_a = int(non_outliers[non_outliers['Fraud_Type'] == 'a']['Time_difference'].mean())
iqr_75 = int(non_outliers['Time_difference'].quantile(0.75))
#mean = int(non_outliers['Time_difference'].mean())
total_df['Time_difference'] = total_df['Time_difference'].apply(lambda x: mean_a if x < 0 else x)
total_df['Time_difference'] = total_df['Time_difference'].apply(lambda x: iqr_75 if x > 700000 else x)

features = ['Customer_credit_rating', 'Customer_flag_change_of_authentications', 'rooting_jailbreak_roaming_VPN', 'Customer_loan_type', 'Customer_flag_terminal_malicious_behaviors', 'Customer_atm_limits', 
            'Account_account_type', 'Account_indicator_release_limit_excess', 'Account_release_suspention',
            'Channel_OS', 'Type_General_Automatic', 'Access_Medium', 'Transaction_num_connection_failure', 'Distance', 'Unused_terminal_status', 'Flag_deposit_more_than_tenMillion', 
            'Unused_account_status', 'Recipient_account_suspend_status', 'Number_of_transaction_with_the_account', 'Transaction_history_with_the_account', 'Transaction_age_group', 
            'Transaction_Amount', 'Account_initial_balance', 'Account_balance', 'Account_one_month_max_amount', 'Account_one_month_std_dev', 'Account_dawn_one_month_max_amount', 'Account_dawn_one_month_std_dev', 'Account_amount_daily_limit', 'Account_remaining_amount_daily_limit_exceeded',
            'Time_difference', 'Time_diff_last_atm_resume', 'Time_diff_last_bank_branch_resume', 'Time_diff_transaction_resume', 'Time_diff_transaction_creation']

categorical_columns = ['Customer_credit_rating', 'Customer_loan_type', 'Account_account_type', 'Channel_OS', 'Type_General_Automatic', 'Access_Medium']

SAMPLE_NUM = 3000
seed = 30
N_CLS_PER_GEN = 40
epochs = 1500

df_m = total_df[total_df["Fraud_Type"] == 'm'].sample(n=SAMPLE_NUM, random_state=seed)
fraud_df = total_df[total_df["Fraud_Type"] != 'm']
train_df = pd.concat([df_m, fraud_df], ignore_index=True)

X = train_df[features].reset_index(drop=True)
y = train_df['Fraud_Type']

fraud_types = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

# 데이터셋 피쳐 메타데이터 생성
with open('metadata.json') as f:
    metadata_dict = json.load(f)

metadata_features = {'columns': {key: metadata_dict['columns'][key] for key in features}}
metadata = SingleTableMetadata.load_from_dict(metadata_features)

def syn_fold(X, y, fold):
    synthetic_data = pd.DataFrame()
    train_df = X.copy()
    train_df['Fraud_Type'] = y

    for fraud_type in tqdm(fraud_types):
        subset = train_df[train_df["Fraud_Type"] == fraud_type][features] 
        synthesizer = CTGANSynthesizer(metadata, epochs=epochs)
        synthesizer.fit(subset)
        with open(f'synthesizer/ctgan_{SAMPLE_NUM}_{seed}_fold{fold}_{fraud_type}.pkl', 'wb') as f:
            pickle.dump(synthesizer, f)
        # 기존에 생성한 synthesizer 파일이 있을 경우 로드
        """
        with open(f'synthesizer/ctgan_{SAMPLE_NUM}_{seed}_fold{fold}_{fraud_type}.pkl', 'rb') as f:
            synthesizer = pickle.load(f)
        """
        synthetic_subset = synthesizer.sample(num_rows=N_CLS_PER_GEN)
        synthetic_subset["Fraud_Type"] = fraud_type
        
        synthetic_data = pd.concat([synthetic_data, synthetic_subset], ignore_index=True)
    
    return synthetic_data


NFOLD = 5
folds = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)
all_synthetic_data = pd.DataFrame()

for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):
    print(f"\n========== Fold {fold + 1} Training Started ==========")

    fold_synthetic_data = syn_fold(X.iloc[train_idx], y[train_idx], fold)
    fold_synthetic_data['fold'] = fold

    print("Fold Synthetic Data Shape:", fold_synthetic_data.shape)

    all_synthetic_data = pd.concat([all_synthetic_data, fold_synthetic_data], ignore_index=True)

print("\n========== Training Complete ==========")
print("Final All Synthetic Data Shape:", all_synthetic_data.shape)
all_synthetic_data.to_csv(f'data/ctgan_{SAMPLE_NUM}_{seed}_{N_CLS_PER_GEN}.csv', encoding='UTF-8-sig', index=False)