import pandas as pd
import numpy as np
import json
from tqdm import tqdm

# 합성 데이터 생성
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# To ignore all warnings
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv("data/train.csv")

total_df = train_df.copy()
total_df.drop("ID", axis=1, inplace=True)
feature_order = total_df.columns.tolist()

total_df['Time_diff_last_atm_resume'] = pd.to_datetime(total_df['Transaction_resumed_date']) - pd.to_datetime(total_df['Last_atm_transaction_datetime'])
total_df['Time_diff_last_bank_branch_resume'] = pd.to_datetime(total_df['Transaction_resumed_date']) - pd.to_datetime(total_df['Last_bank_branch_transaction_datetime'])
total_df['Time_diff_transaction_resume'] = pd.to_datetime(total_df['Transaction_Datetime']) - pd.to_datetime(total_df['Transaction_resumed_date'])
total_df['Time_diff_transaction_creation'] = pd.to_datetime(total_df['Transaction_Datetime']) - pd.to_datetime(total_df['Account_creation_datetime'])
total_df['Time_diff_creation_registreation'] = pd.to_datetime(total_df['Account_creation_datetime']) - pd.to_datetime(total_df['Customer_registration_datetime'])
total_df['Transaction_Age'] = pd.to_datetime(total_df['Transaction_Datetime']).dt.year - total_df['Customer_Birthyear']

datetime_columns = ['Time_difference', 'Time_diff_last_atm_resume', 'Time_diff_last_bank_branch_resume', 'Time_diff_transaction_resume', 'Time_diff_transaction_creation', 'Time_diff_creation_registreation']
for col in datetime_columns:
    total_df[col] = pd.to_timedelta(total_df[col], errors='coerce').dt.total_seconds()
    total_df[col] = total_df[col].fillna(0)

total_df['Last_Name'] = total_df['Customer_personal_identifier'].str[0] 
total_df['First_Name'] = total_df['Customer_personal_identifier'].str[1:]
total_df[['Cid1', 'Cid2']] = total_df['Customer_identification_number'].str.split('-', expand=True)
total_df['Channel_OS'] = total_df['Channel'] + "_" + total_df["Operating_System"]

# outlier 처리
non_outliers = total_df[(total_df['Time_difference'] >= 0) & (total_df['Time_difference'] < 700000)]
mean_a = int(non_outliers[non_outliers['Fraud_Type'] == 'a']['Time_difference'].mean())
iqr_75 = int(non_outliers['Time_difference'].quantile(0.75))
total_df['Time_difference'] = total_df['Time_difference'].apply(lambda x: mean_a if x < 0 else x)
total_df['Time_difference'] = total_df['Time_difference'].apply(lambda x: iqr_75 if x > 700000 else x)

features = ['Customer_Birthyear', 'Transaction_Age', 'Customer_Gender', 'Customer_credit_rating', 'Customer_flag_change_of_authentication_1', 'Customer_flag_change_of_authentication_2', 'Customer_flag_change_of_authentication_3', 'Customer_flag_change_of_authentication_4',
            'Customer_rooting_jailbreak_indicator', 'Customer_mobile_roaming_indicator', 'Customer_VPN_Indicator', 'Customer_loan_type', 
            'Customer_flag_terminal_malicious_behavior_1', 'Customer_flag_terminal_malicious_behavior_2', 'Customer_flag_terminal_malicious_behavior_3', 'Customer_flag_terminal_malicious_behavior_4', 'Customer_flag_terminal_malicious_behavior_5', 'Customer_flag_terminal_malicious_behavior_6', 
            'Customer_inquery_atm_limit', 'Customer_increase_atm_limit', 'Account_account_type', 'Account_indicator_release_limit_excess',  'Account_indicator_Openbanking', 'Account_release_suspention', 
            'Channel_OS', 'Error_Code', 'Transaction_Failure_Status', 'Type_General_Automatic', 'Access_Medium', 'Transaction_num_connection_failure', 'Another_Person_Account', 'Distance', 
            'Unused_terminal_status', 'Flag_deposit_more_than_tenMillion', 'Unused_account_status', 'Recipient_account_suspend_status', 'Number_of_transaction_with_the_account', 'Transaction_history_with_the_account', 'First_time_iOS_by_vulnerable_user', 
            'Transaction_Datetime', 'Time_difference', 'Time_diff_last_atm_resume', 'Time_diff_last_bank_branch_resume', 'Time_diff_transaction_resume', 'Time_diff_transaction_creation', 'Time_diff_creation_registreation',
            'Transaction_Amount', 'Account_initial_balance', 'Account_balance', 'Account_one_month_max_amount', 'Account_one_month_std_dev', 'Account_dawn_one_month_max_amount', 'Account_dawn_one_month_std_dev', 'Account_amount_daily_limit', 'Account_remaining_amount_daily_limit_exceeded',
            'Account_account_number', 'Recipient_Account_Number', 'IP_Address', 'MAC_Address', 'Location', 
            'First_Name', 'Last_Name', 'Cid1', 'Cid2']

annomaly_features = ['First_Name', 'Last_Name', 'Cid1', 'Cid2', 'Account_account_number', 'Recipient_Account_Number', 'Location']

N_CLS_PER_GEN = 1000
fraud_types = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']

# 메타데이터 생성
with open('metadata.json') as f:
    metadata_dict = json.load(f)

metadata_features = {'columns': {key: metadata_dict['columns'][key] for key in features}}
metadata = SingleTableMetadata.load_from_dict(metadata_features)

syn_df = pd.DataFrame()
for fraud_type in tqdm(fraud_types):
    subset = total_df[total_df["Fraud_Type"] == fraud_type][features]
    if fraud_type == 'm':
        subset = subset.sample(n=1000, random_state=42)
        
    synthesizer = CTGANSynthesizer(metadata, epochs=100)
    synthesizer.fit(subset)

    synthetic_subset = synthesizer.sample(num_rows=N_CLS_PER_GEN)
    synthetic_subset['Fraud_Type'] = fraud_type

    syn_df = pd.concat([syn_df, synthetic_subset], ignore_index=True)
    
syn_df['Customer_personal_identifier'] = syn_df['Last_Name'] + syn_df['First_Name']
syn_df['Customer_identification_number'] = syn_df['Cid1'] + "-" + syn_df['Cid2']
syn_df[['Channel', 'Operating_System']] = syn_df['Channel_OS'].str.split('_', expand=True)

syn_df['Time_difference'] = pd.to_timedelta(syn_df['Time_difference'], unit='s')
syn_df['Account_creation_datetime'] = pd.to_datetime(syn_df['Transaction_Datetime']) - pd.to_timedelta(syn_df['Time_diff_transaction_creation'], unit='s')
syn_df['Customer_registration_datetime'] = pd.to_datetime(syn_df['Account_creation_datetime']) - pd.to_timedelta(syn_df['Time_diff_creation_registreation'], unit='s')
syn_df['Transaction_resumed_date'] = pd.to_datetime(syn_df['Transaction_Datetime']) - pd.to_timedelta(syn_df['Time_diff_transaction_resume'], unit='s')
syn_df['Last_atm_transaction_datetime'] = pd.to_datetime(syn_df['Transaction_resumed_date']) - pd.to_timedelta(syn_df['Time_diff_last_atm_resume'], unit='s')
syn_df['Last_bank_branch_transaction_datetime'] = pd.to_datetime(syn_df['Transaction_resumed_date']) - pd.to_timedelta(syn_df['Time_diff_last_bank_branch_resume'], unit='s')
#syn_df['Customer_Birthyear'] = pd.to_datetime(syn_df['Transaction_Datetime']).dt.year - syn_df['Transaction_Age']

datetimes = ['Time_difference', 'Account_creation_datetime', 'Customer_registration_datetime', 'Transaction_resumed_date', 'Last_atm_transaction_datetime', 'Last_bank_branch_transaction_datetime']
for i in datetimes:
    syn_df[i] = syn_df[i].astype(str)
    
syn_df.to_csv('submission/syn_submission.csv', encoding='UTF-8-sig', index=False)