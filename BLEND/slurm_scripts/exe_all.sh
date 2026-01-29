#!/usr/bin/env bash
set -euo pipefail

# ==== choose ONE subsample ====
SUBSAMPLE_CLASS="base"   # or "new"

# ==== extra CLI parameters ====
ALPHAS=(0.01)
NUM_USERS=10
NUM_GLOBAL_ROUNDS=50
NUM_SHOTS=16
GAMMA_IMPORTANCE=0.09
GAMMA2=0.2
ADPTER_DIMS=256
# ==== config ====
ROOT="BLEND"
TRAINER="BLEND"
DATASETS=(
  # imagenet
  caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101
)
SEEDS=(1 2)
# TIDS=(0 5 10 20 40 80 100)

SIF="pytorch.sif"
PY="BLEND/federated_main.py"
DS_CFG_DIR="${ROOT}/configs/datasets"
TR_CFG="BLEND/configs/trainers/BLEND/vit_b16_ep5.yaml"
# ${ROOT}/configs/trainers/Fed_MMRL/vit_b16.yaml
# slurm filesystem layout roots
SCRIPTS_ROOT="${ROOT}/slurms/scripts"
LOGS_ROOT="${ROOT}/slurms/logs"


JOB_PREFIX="train_gamma2"

for ALPHA in "${ALPHAS[@]}"; do
  ALPHA_TAG="${ALPHA//./p}"  # 0.01 -> 0p01 for file/dir names
  # for tid in "${TIDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      # --- directories for scripts/logs: alpha -> seed -> trainer -> dataset -> subsample ---
      script_dir="${SCRIPTS_ROOT}/alpha${ALPHA_TAG}/seed${seed}/${TRAINER}/${ds}/${SUBSAMPLE_CLASS}"
      log_dir="${LOGS_ROOT}/alpha${ALPHA_TAG}/seed${seed}/${TRAINER}/${ds}/${SUBSAMPLE_CLASS}"
      mkdir -p "${script_dir}" "${log_dir}"

      # --- output-dir (NO subsample class here) ---
      output_dir="${ROOT}/output/${ALPHA}/${seed}/${TRAINER}/${ds}"
      mkdir -p "${output_dir}"

      job="${JOB_PREFIX}_a${ALPHA_TAG}_s${seed}_${TRAINER}_${ds}_${SUBSAMPLE_CLASS}"
      slurm_file="${script_dir}/${job}.slurm"
      out_file="${log_dir}/${ds}_${SUBSAMPLE_CLASS}_a${ALPHA_TAG}_s${seed}.%j.out"

      cmd="fuser -k 5678/tcp 2>/dev/null || true; apptainer exec --nv ${SIF} python '${PY}' \
        --trainer '${TRAINER}' \
        --dataset-config-file '${DS_CFG_DIR}/${ds}.yaml' \
        --subsample '${SUBSAMPLE_CLASS}' \
        --config-file '${TR_CFG}' \
        --seed ${seed} \
        --dir_alpha ${ALPHA} \
        --num_users ${NUM_USERS} \
        --num_shots ${NUM_SHOTS} \
        --gamma_importance ${GAMMA_IMPORTANCE} \
        --gamma2 ${GAMMA2}\
        --adapter_dim 16\
        --per_lora_dim 256\
        --output_dir '${output_dir}'"

      cat > "${slurm_file}" <<SLURM
#!/bin/bash
#
#SBATCH --time=06:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --job-name=${job}
#SBATCH --output=${out_file}
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=ALL
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=alipour
#SBATCH --cluster=ub-hpc
#SBATCH --gpus-per-node=1
#SBATCH --constraint=A100

${cmd}
SLURM

        # sbatch "${slurm_file}"
      echo "Submitted: ${slurm_file} -> ${out_file}"
    done
  done 
done
