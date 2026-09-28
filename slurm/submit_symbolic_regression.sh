#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_PROJECT="${JULIA_PROJECT:-$ROOT/julia}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
PARTITION="${SLURM_PARTITION:-normal}"
QOS="${SLURM_QOS:-normal}"
DATA_DIR="${TI_ALLOY_DATA_DIR:-$ROOT/outputs/data}"
LOG_DIR="${TI_ALLOY_LOG_DIR:-$ROOT/outputs/logs}"
SCRIPT="$ROOT/julia/symbolic_regression_search.jl"
mkdir -p "$LOG_DIR"
for TARGET in yield_strength_mpa ultimate_tensile_strength_mpa; do
 for POP in 20 40 60; do
  for PAR in 0.00032 0.0032 0.032; do
   for MAX in 8 12 16; do
    TAG="pop${POP}_par${PAR}_max${MAX}";JOB="$LOG_DIR/sub_${TARGET}_${TAG}.sh"
    cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH -J sr_${TARGET}_${TAG}
#SBATCH -o ${LOG_DIR}/sr_${TARGET}_${TAG}.out
#SBATCH -e ${LOG_DIR}/sr_${TARGET}_${TAG}.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH -n 8
export JULIA_PKG_OFFLINE=true SR_POPULATIONS=${POP} SR_PARSIMONY=${PAR} SR_MAXSIZE=${MAX} SR_NITER=2000
"${JULIA_BIN}" --project="${JULIA_PROJECT}" --threads=8 --startup-file=no "${SCRIPT}" "${TARGET}" "${DATA_DIR}" "${TAG}"
EOF
    "$SBATCH_BIN" "$JOB"
   done
  done
 done
done

JOB="$LOG_DIR/sub_elongation_pct_el_single.sh"
cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH -J sr_el_single
#SBATCH -o ${LOG_DIR}/sr_el_single.out
#SBATCH -e ${LOG_DIR}/sr_el_single.err
#SBATCH --partition=${PARTITION}
#SBATCH --qos=${QOS}
#SBATCH -n 8
export JULIA_PKG_OFFLINE=true SR_POPULATIONS=33 SR_PARSIMONY=0.0032 SR_MAXSIZE=12 SR_NITER=2000
"${JULIA_BIN}" --project="${JULIA_PROJECT}" --threads=8 --startup-file=no "${SCRIPT}" elongation_pct "${DATA_DIR}" el_single
EOF
"$SBATCH_BIN" "$JOB"
