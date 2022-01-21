#MAIN_FILE=/home/mehran/Documents/pytorch-soft-actor-critic/main.py
MAIN_FILE=/local/melco3/taghianj/pytorch-soft-actor-critic/main.py


tmux new-session -d -s v0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name AntEnv-v1 --automatic_entropy_tuning True --num_steps 10000000 --start_steps 10000 --cuda --seed 0"

#python main.py --env-name AntEnv-v1 --automatic_entropy_tuning True --num_steps 10000000 --cuda --seed 0