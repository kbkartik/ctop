#!/bin/bash

# Script to reproduce results

export CUDA_VISIBLE_DEVICES=1

for i in 46
do
    pseed=$((i+1))
    vseed=$i
    fversion="_navg"
    codename="ctop_v4"

    numactl -C 20 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Hopper-v2" --seed $pseed --filename $codename"_h_s"$pseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_h_s"$pseed"_lr_1e1"$fversion".txt" &

    numactl -C 21 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Walker2d-v2" --seed $pseed --filename $codename"_w_s"$pseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_w_s"$pseed"_lr_1e1"$fversion".txt" &

    numactl -C 22 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "HalfCheetah-v2" --seed $pseed --filename $codename"_hc_s"$pseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_hc_s"$pseed"_lr_1e1"$fversion".txt"
done