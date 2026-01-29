#!/usr/bin/env bash
set -euo pipefail

# ==== choose ONE subsample ====
SUBSAMPLE_CLASS="new"   # or "new"

# ==== extra CLI parameters ====
ALPHAS=(0.01)
NUM_USERS=10
NUM_GLOBAL_ROUNDS=50
NUM_SHOTS=16

# ==== config ====
ROOT="FedPHA-main-test"
TRAINER="PFEDMOAP"
DATASETS=(
  caltech101 dtd eurosat fgvc_aircraft food101
  oxford_flowers oxford_pets stanford_cars ucf101
)
SEEDS=(1 2 3)

SIF="pytorch.sif"
PY="FedPHA-main-test/test.py"
DS_CFG_DIR="${ROOT}/configs/datasets"
TR_CFG="FedPHA-main-test/configs/trainers/PFEDMOAP/vit_b16.yaml"
# ${ROOT}/configs/trainers/Fed_MMRL/vit_b16.yaml
# slurm filesystem layout roots
SCRIPTS_ROOT="${ROOT}/slurms_test_gen/scripts"
LOGS_ROOT="${ROOT}/slurms_test_gen/logs"

# Determine eval-only vs train based on SUBSAMPLE_CLASS

JOB_PREFIX="g_test"

for ALPHA in "${ALPHAS[@]}"; do
  ALPHA_TAG="${ALPHA//./p}"  # 0.01 -> 0p01 for file/dir names

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
      
      #--personalized_test\
      cmd="apptainer exec --nv ${SIF} python '${PY}' \
        --trainer '${TRAINER}' \
        --dataset-config-file '${DS_CFG_DIR}/${ds}.yaml' \
        --subsample '${SUBSAMPLE_CLASS}' \
        --config-file '${TR_CFG}' \
        --seed ${seed} \
        --dir_alpha ${ALPHA} \
        --num_users ${NUM_USERS} \
        --num_shots ${NUM_SHOTS} \
        --output_dir '${output_dir}'"

      cat > "${slurm_file}" <<SLURM
#!/bin/bash
#
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=10000
#SBATCH --job-name=${job}
#SBATCH --output=${out_file}
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=ALL
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=mikel
#SBATCH --cluster=ub-hpc
#SBATCH --gpus-per-node=1
#SBATCH --constraint=V100

${cmd}
SLURM

      sbatch "${slurm_file}"
      echo "Submitted: ${slurm_file} -> ${out_file}"
    done
  done
done
