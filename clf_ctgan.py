import pandas as pd
import numpy as np

from time import time
import logging
import pickle

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

import lightgbm as lgb

# To ignore all warnings
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(filename='log/output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

total_df = pd.concat([train_df, test_df], axis=0, sort=False)
total_df.drop("ID", axis=1, inplace=True)

total_df['Time_diff_last_atm_resume'] = pd.to_datetime(total_df['Transaction_resumed_date']) - pd.to_datetime(total_df['Last_atm_transaction_datetime'])
total_df['Time_diff_last_bank_branch_resume'] = pd.to_datetime(total_df['Transaction_resumed_date']) - pd.to_datetime(total_df['Last_bank_branch_transaction_datetime'])
total_df['Time_diff_transaction_resume'] = pd.to_datetime(total_df['Transaction_Datetime']) - pd.to_datetime(total_df['Transaction_resumed_date'])
total_df['Time_diff_transaction_creation'] = pd.to_datetime(total_df['Transaction_Datetime']) - pd.to_datetime(total_df['Account_creation_datetime'])

# datetime 변환
datetime_columns = ['Time_difference', 'Time_diff_last_atm_resume', 'Time_diff_last_bank_branch_resume', 'Time_diff_transaction_resume', 'Time_diff_transaction_creation']
for col in datetime_columns:
    total_df[col] = pd.to_timedelta(total_df[col], errors='coerce').dt.total_seconds()
    total_df[col] = total_df[col].fillna(0)
    
total_df['Transaction_Age'] = pd.to_datetime(total_df['Transaction_Datetime']).dt.year - total_df['Customer_Birthyear']
total_df['Transaction_age_group'] = pd.cut(total_df['Transaction_Age'], bins=[-1, 19, 29, 39, 49, 59, 99], labels=[0, 1, 2, 3, 4, 5])
    
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
            'Time_difference', 'Time_diff_last_atm_resume', 'Time_diff_last_bank_branch_resume', 'Time_diff_transaction_resume', 'Time_diff_transaction_creation',
            'ChannelOS_Account_initial_balance_mean', 'Account_account_type_Account_initial_balance_min']

categorical_columns = ['Customer_credit_rating', 'Customer_loan_type', 'Account_account_type', 'Channel_OS', 'Type_General_Automatic', 'Access_Medium']

train_df = total_df[total_df['Fraud_Type'].notnull()]
test_df = total_df[total_df['Fraud_Type'].isnull()]

NUM_M = 3000
seed = 30
N_CLS_PER_GEN = 40

df_m = train_df[train_df["Fraud_Type"] == 'm'].sample(n=NUM_M, random_state=seed)
fraud_df = train_df[train_df["Fraud_Type"] != 'm']
trainsp_df = pd.concat([df_m, fraud_df], ignore_index=True)

X = trainsp_df[features].reset_index(drop=True)
y = trainsp_df['Fraud_Type']

test_X = test_df[features].reset_index(drop=True)

# 라벨 인코딩
le = LabelEncoder()
y = le.fit_transform(y)
y = pd.Series(y)

# 범주형 변수 인코딩
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X[categorical_columns] = oe.fit_transform(X[categorical_columns])
feature_order = X.columns.tolist()

test_X[categorical_columns] = oe.transform(test_X[categorical_columns])
test_X = test_X[feature_order]

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

best_params = {'learning_rate': 0.005, 'num_leaves': 31, 'max_depth': -1, 'min_data_in_leaf': 20}
params = {'objective': 'multiclass', 'num_class': 13, 'metric': 'multi_logloss', 'boosting_type': 'gbdt', 'seed': 42, 'verbose': -1}
params.update(best_params)

NFOLD = 5
folds = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)

best_fold_f1 = {fold: {'f1': 0, 'i': 0} for fold in range(5)}
    
for i in range(3):
    print(f"\n========== CTGAN {i} Training Started ==========")
    syn_df = pd.DataFrame()
    oofs = np.zeros(len(trainsp_df))
    preds = np.zeros((len(test_df), 13))

    training_start_time = time()
    for fold, (train_idx, test_idx) in enumerate(folds.split(X, y)):
        fold_start_time = time()
        print(f"\n========== Fold {fold + 1} Training Started ==========")
        
        # CTGAN 기법으로 데이터 생성(syn_ctgan.py)
        fold_syn_df = pd.DataFrame()
        for label in labels:
            with open(f'synthesizer/ctgan_{NUM_M}_{seed}_fold{fold}_{label}.pkl', 'rb') as f:
                synthesizer = pickle.load(f)
            synthetic_subset = synthesizer.sample(num_rows=N_CLS_PER_GEN)
            synthetic_subset["Fraud_Type"] = label
        
            fold_syn_df = pd.concat([fold_syn_df, synthetic_subset], ignore_index=True)
            
        fold_syn_df['fold'] = fold
        syn_df = pd.concat([syn_df, fold_syn_df], ignore_index=True)
                
        # Channel_OS별 Account_initial_balance의 평균 계산
        channel_os_mean = total_df.groupby('Channel_OS')['Account_initial_balance'].mean().reset_index()
        channel_os_mean.columns = ['Channel_OS', 'ChannelOS_Account_initial_balance_mean']
        fold_syn_df = fold_syn_df.merge(channel_os_mean, on='Channel_OS', how='left')

        # Account_account_type별 Account_initial_balance의 최소값 계산
        account_type_min = total_df.groupby('Account_account_type')['Account_initial_balance'].min().reset_index()
        account_type_min.columns = ['Account_account_type', 'Account_account_type_Account_initial_balance_min']
        fold_syn_df = fold_syn_df.merge(account_type_min, on='Account_account_type', how='left')
        
        syn_X = fold_syn_df[features].reset_index(drop=True)
        syn_y = fold_syn_df['Fraud_Type']
        
        syn_y = le.transform(syn_y)
        syn_y = pd.Series(syn_y)
        
        syn_X[categorical_columns] = oe.transform(syn_X[categorical_columns])
        syn_X = syn_X[feature_order]
        for col in feature_order:
            syn_X[col] = syn_X[col].astype(X[col].dtype)
            
        fold_X = pd.concat([X.iloc[train_idx], syn_X], axis=0, sort=False)
        fold_X = fold_X.reset_index(drop=True)
        fold_y = pd.concat([y.iloc[train_idx], syn_y], axis=0, sort=False)
        
        train_data = lgb.Dataset(fold_X, label=fold_y)
        val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
        
        # 학습 시작
        clf = lgb.train(params, train_data, 3000, valid_sets=[train_data, val_data], verbose_eval=False)
        oofs[test_idx] = np.argmax(clf.predict(X.iloc[test_idx]), axis=1)
        f1 = f1_score(y.iloc[test_idx], oofs[test_idx], average='macro')
        #preds += clf.predict(test_X) / NFOLD
        
        print(f"Validation F1 Score (Macro): {f1 * 100:.2f}%")
        
        # 폴드별로 최고 f1 점수 업데이트
        if f1 > best_fold_f1[fold]['f1']:
            best_fold_f1[fold]['f1'] = f1
            best_fold_f1[fold]['i'] = i
            print(f"New best F1 for fold {fold + 1}: {f1 * 100:.2f}% at i = {i}")
        
        report = classification_report(y.iloc[test_idx], oofs[test_idx], target_names=[str(label) for label in range(13)])
        logging.info(f"========== Fold {fold + 1}  ==========")
        logging.info(f"\n{report}")
        logging.info(f"Validation F1 Score (Macro): {f1_score(y.iloc[test_idx], oofs[test_idx], average='macro') * 100:.2f}%")
        
    print("\n========== Training Complete ==========")
    print(f"Overall F1 Score (Macro): {f1_score(y, oofs, average='macro') * 100:.2f}%")
    logging.info("========== Training Complete ==========")
    logging.info(f"Overall F1 Score (Macro): {f1_score(y, oofs, average='macro') * 100:.2f}%")

    #predictions_label = le.inverse_transform(np.argmax(preds, axis=1))
    #print(np.unique(predictions_label, return_counts=True))
    #logging.info(np.unique(predictions_label, return_counts=True))

    #submission = pd.read_csv("data/sample_submission.csv")
    #submission['Fraud_Type'] = predictions_label
    #submission.to_csv(f'submission/clf_submission_{NUM_M}_{seed}_{N_CLS_PER_GEN}_auto_{i}.csv', index=False)
    
    syn_df.to_csv(f'data/ctgan_{NUM_M}_{seed}_{N_CLS_PER_GEN}_auto_{i}.csv', encoding='UTF-8-sig', index=False)
    
for fold, info in best_fold_f1.items():
    print(f"Fold {fold + 1}: Best F1 = {info['f1'] * 100:.2f}% at i = {info['i']}")
