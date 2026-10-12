#!/usr/bin/env bash
# Step 0 of ENCODER_PLAN.md: all nine runs. The three conditions run side by side, each over seeds 0, 1, 2 one after
# another (about 10 minutes per run). Run it from the activated carlos env; logs go to runs_step0/0a.log, 0b.log, 0c.log.
#
#   conda activate carlos
#   nohup bash run_step0.sh > run_step0.log 2>&1 &      # keeps running if the SSH session ends
set -u
cd "$(dirname "$0")"
mkdir -p runs_step0
# Two CPU threads per run, so the three runs share the machine
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python ppo/main_battery_plane_ppo.py -m +experiment=S0a_observed_bs256 training.seed=0,1,2 > runs_step0/0a.log 2>&1 &
pid_0a=$!
python ppo/main_battery_plane_ppo.py -m +experiment=S0b_observed_bs1024 training.seed=0,1,2 > runs_step0/0b.log 2>&1 &
pid_0b=$!
python main_battery_plane_contextual_ppo.py -m +step0=cppo_oracle training.seed=0,1,2 training.num_threads=2 \
    > runs_step0/0c.log 2>&1 &
pid_0c=$!
echo "Started 0a, 0b and 0c; logs in runs_step0/"

status=0
for run in 0a 0b 0c; do
    pid_var="pid_$run"
    if wait "${!pid_var}"; then
        echo "$run finished"
    else
        echo "$run FAILED: see runs_step0/$run.log"
        status=1
    fi
done
exit $status
