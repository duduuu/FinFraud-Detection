# 2nd Place Solution for FSI AIxData Challenge 2024
This repository contains the code for the 2nd place solution in the FSI AIxData Challenge 2024, achieving a private score of 0.81983.

- Team Name: 인형눈붙이기장인
- Team Members: 백두현(두둡), 신찬호(인형눈붙이기장인)

# Repository Structure

```bash
📦FSI_AIxDATA
 ┣ 📂data                              # Dataset and generated synthetic data
 ┃ ┣ 📜ctgan_3000_30_40_auto_0.csv
 ┃ ┣ 📜...
 ┃ ┣ 📜ctgan_3600_50_80_auto_10.csv
 ┃ ┣ 📜train.csv
 ┃ ┣ 📜test.csv
 ┃ ┗ 📜sample_submission.csv
 ┣ 📂log                               # Training logs and feature importance data
 ┃ ┣ 📜feature_importances_lgb.csv
 ┃ ┣ ...
 ┃ ┣ 📜feature_importances_single_l.csv
 ┃ ┣ 📜output.log
 ┃ ┗ 📜output_ctgan.log
 ┣ 📂synthesizer                       # Trained synthetic models
 ┃ ┣ 📜ctgan_3000_30_fold0_a.pkl
 ┃ ┣ ...
 ┃ ┗ 📜ctgan_3600_50_fold4_l.pkl
 ┣ 📂submission                        # Submission files
 ┃ ┣ 📜clf_submission_3000_30_40_ctgan.csv
 ┃ ┣ ...
 ┃ ┣ 📜clf_submission_3600_50_ctgan.csv
 ┃ ┣ 📜clf_submission.csv
 ┃ ┗ 📜syn_submission.csv
 ┣ 📜README.md
 ┣ 📜EDA.ipynb                         # Exploratory data analysis
 ┣ 📜syn.py                            # synthetic data generation script
 ┣ 📜syn_ctgan.py                      # Training synthetic data generation script
 ┣ 📜clf.ipynb                         # Classification model training script
 ┣ 📜clf_ctgan.py                      # Main classification model training script
 ┣ 📜clf_best.py                       # Best-performing model script
 ┣ 📜ensemble.py                       # Ensemble method for combining models
 ┣ 📜metadata.json                     # Metadata for the synthetic models
 ┗ 📜requirements.txt                  
```

# How to Use
  **주의사항 : CTGAN은 데이터 생성 시 seed 설정 기능이 없어 생성 결과를 대신 저장했기 때문에 결과 검증 시에는 clf_best.py부터 진행해야 합니다.**
  - **Train synthetic models for submission : syn.py**  
    
  - **Train synthetic models for data augmentation: : syn_ctgan.py**
    ```bash
    SAMPLE_NUM = 3000
    seed = 30
    N_CLS_PER_GEN = 40
    epochs = 1500
    ```
    The trained models will be saved in the synthesizer folder with the following format:
    ```bash
    synthesizer/ctgan_{SAMPLE_NUM}_{seed}_fold{fold}_{fraud_type}.pkl
    ```
    
  - **Search best classification model  : clf_ctgan.py**  
    ```bash
    NUM_M = 3000
    seed = 30
    N_CLS_PER_GEN = 40
    ```
    The generated synthetic data will be saved in the data/ folder, and the training logs will be saved as output.log.  
    ```bash
    data/ctgan_{NUM_M}_{seed}_{N_CLS_PER_GEN}_auto_{i}.csv
    log/output.log
    ```

  - **Train the best classification model  : clf_best.py**
    
    selecting the best synthetic data for the classification model.
    ```bash
    best_clf(NUM_M = 3000, seed = 30, N_CLS_PER_GEN = 40, best_syn_df = [-1, 3, -1, 4, 0])
    ```
    The resulting file will be saved in the submission/ folder:
    ```bash
    submission/clf_submission_{NUM_M}_{seed}_{N_CLS_PER_GEN}_ctgan.csv
    ```
  
  - **Model Ensembling : ensemble.py**

# Result
  - **2nd Place** : Achieved a private score of 0.81983.
