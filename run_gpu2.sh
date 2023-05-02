#!/bin/bash

# Script to reproduce results

export CUDA_VISIBLE_DEVICES=1

for i in 46
do
    pseed=$((i+3))
    p2seed=$((i+4))
    vseed=$i
    fversion="_navg"
    codename="ctop_v4"

    numactl -C 23 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Hopper-v2" --seed $pseed --filename $codename"_h_s"$pseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_h_s"$pseed"_lr_1e1"$fversion".txt" &

    numactl -C 24 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Walker2d-v2" --seed $pseed --filename $codename"_w_s"$pseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_w_s"$pseed"_lr_1e1"$fversion".txt" &

    numactl -C 25 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "HalfCheetah-v2" --seed $pseed --filename $codename"_hc_s"$pseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_hc_s"$pseed"_lr_1e1"$fversion".txt" &

    numactl -C 26 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Walker2d-v2" --seed $vseed --filename $codename"_w_s"$vseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_w_s"$vseed"_lr_1e1"$fversion".txt" &

    numactl -C 27 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Hopper-v2" --seed $p2seed --filename $codename"_h_s"$p2seed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_h_s"$p2seed"_lr_1e1"$fversion".txt" &

    numactl -C 28 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "Walker2d-v2" --seed $p2seed --filename $codename"_w_s"$p2seed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_w_s"$p2seed"_lr_1e1"$fversion".txt" &

    numactl -C 29 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "HalfCheetah-v2" --seed $p2seed --filename $codename"_hc_s"$p2seed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_hc_s"$p2seed"_lr_1e1"$fversion".txt" &

    numactl -C 30 -m 1 python -u ./mujoco/ctop_agent_spsa.py --env "HalfCheetah-v2" --seed $vseed --filename $codename"_hc_s"$vseed"_lr_1e1"$fversion --bandit_lr 0.1 --fb_type "subfb_wolrd" > ./results/$codename"_hc_s"$vseed"_lr_1e1"$fversion".txt"
done