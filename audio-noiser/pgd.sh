#! /bin/bash
SEED=100
NBITER=1
LOAD=True
DATA=data/test-clean-100
NAME=pgd
SEED=235

SNR=35
LAMBDA=100
python3 run_attack.py attack_configs/whisper/pgd.yaml --root="MLSP_Attack" --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --lambda_stoi=$LAMBDA
LAMBDA=50
python3 run_attack.py attack_configs/whisper/pgd.yaml --root="MLSP_Attack" --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --lambda_stoi=$LAMBDA
LAMBDA=10
python3 run_attack.py attack_configs/whisper/pgd.yaml --root="MLSP_Attack" --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --lambda_stoi=$LAMBDA

SNR=30
LAMBDA=100
python3 run_attack.py attack_configs/whisper/pgd.yaml --root="MLSP_Attack" --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --lambda_stoi=$LAMBDA
LAMBDA=50
python3 run_attack.py attack_configs/whisper/pgd.yaml --root="MLSP_Attack" --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --lambda_stoi=$LAMBDA
LAMBDA=10
python3 run_attack.py attack_configs/whisper/pgd.yaml --root="MLSP_Attack" --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --attack_name=$NAME --snr=$SNR --lambda_stoi=$LAMBDA