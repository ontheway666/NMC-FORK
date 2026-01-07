#!/bin/bash\
vis_resolution=256

time python3 ../../src/2d/main.py \
    --src_dir ../../src/2d \
    --src karmanEmpty \
    --exp_name karmanEmpty_vertVel_4edgeEmpty\
    --ckpt -1 \
    --lr 1e-5 \
    --network siren \
    --nonlinearity sine \
    --num_hidden_layers 2 \
    --hidden_features 128 \
    --dt 0.05 \
    --time_integration semi_lag \
    --early_stop \
    --max_n_iters 10000 \
    --n_timesteps 500 \
    --sample random \
    --sample_resolution 128 \
    --vis_resolution $vis_resolution \
    --vel_vis_resolution 200 \
    --wost_resolution 512 \
    --fps 10 \
    --bdry_eps 3e-2 \
    --karman_vel 2 \
    --wost_json ./wost_4edgeEmpty.json \
    --adv_ref 0 \
    --reset_wts 1

python3 plot_scalar.py ../../results/karman/results/txt/ $vis_resolution
