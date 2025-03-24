# CUDA_VISIBLE_DEVICES=0 python run.py --graph_size 20 --baseline rollout --run_name 'tsp20_rollout' --no_tensorboard

# CUDA_VISIBLE_DEVICES=1 python run.py --graph_size 50 --baseline rollout --run_name 'tsp50_rollout' --no_tensorboard

CUDA_VISIBLE_DEVICES=2 python run.py --graph_size 100 --baseline rollout --run_name 'tsp100_rollout' --no_tensorboard

# CUDA_VISIBLE_DEVICES=3 python run.py --graph_size 200 --baseline rollout --run_name 'tsp200_rollout' --no_tensorboard