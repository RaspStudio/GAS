#!/usr/bin/env bash
# ============================================================
#  GAS WIT Example — auto-detect artifacts, skip if present
# ============================================================
#  Usage:
#    ./run_wit.sh                                        # auto-download + run
#    ./run_wit.sh --wit-dir /path/to/existing_parquets    # use existing parquets
#
#  Requires: cmake, make, g++, python3, wget, pyarrow, numpy
#  Data: WIT (wikimedia/wit_base) first 6 parquet parts
#        100k base + 1k query, ~4% selectivity range filter
# ============================================================
set -euo pipefail

# -------------------- Config --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
TEMP_DIR="$SCRIPT_DIR/temp"
CACHE_DIR="$TEMP_DIR/cache"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
WIT_DIR="$TEMP_DIR/wit_parquets"

BASE_NAME="wit_base.fvecs"
QUERY_NAME="wit_query.fvecs"
BMETA_NAME="wit_base.bmeta"
QMETA_NAME="wit_query_range.qmeta"

DIM=2048
MAX_ELEMENTS=100000
MAX_QUERIES=1000
NUM_PARTS=6
K=10

HF_ENDPOINT="https://huggingface.co"
# HF_ENDPOINT="https://hf-mirror.com" # only if you have issues with the original server
WIT_DATASET="wikimedia/wit_base"
WIT_PATTERN="train-%05d-of-00330.parquet"

ONLY_RUN_IDX=7
REPEAT=1
N_SEG=1
BATCH_SIZE=16
QUERY_SEQ_MODE="normal"
EFS=(10 15 20 30)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# -------------------- Parse args (only --wit-dir) --------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wit-dir) WIT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--wit-dir PATH]"
            echo "  --wit-dir PATH   path to existing WIT parquet directory (skip download)"
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ============================================================
# Dependency check
# ============================================================
check_deps() {
    local missing=() py_missing=()
    for cmd in cmake make g++ python3; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        missing+=("wget or curl")
    fi
    for mod in pyarrow numpy; do
        python3 -c "import $mod" &>/dev/null || py_missing+=("$mod")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo -e "${RED}Missing system deps: ${missing[*]}${NC}"
        echo "   Install: apt install ${missing[*]}"
    fi
    if [[ ${#py_missing[@]} -gt 0 ]]; then
        echo -e "${RED}Missing Python modules: ${py_missing[*]}${NC}"
        echo "   Install: pip install ${py_missing[*]}"
    fi
    [[ ${#missing[@]} -eq 0 && ${#py_missing[@]} -eq 0 ]] || exit 1
    echo -e "${GREEN}All dependencies satisfied${NC}"
}

# ============================================================
# Artifact detection helpers
# ============================================================
all_parquets_exist() {
    for i in $(seq 0 $((NUM_PARTS - 1))); do
        local f="$WIT_DIR/$(printf "$WIT_PATTERN" "$i")"
        [[ -f "$f" ]] || return 1
    done
    return 0
}

extract_outputs_exist() {
    for f in "$TEMP_DIR/$BASE_NAME" "$TEMP_DIR/$BMETA_NAME" \
             "$TEMP_DIR/$QUERY_NAME" "$TEMP_DIR/$QMETA_NAME"; do
        [[ -f "$f" ]] || return 1
    done
    return 0
}

# ============================================================
# Main
# ============================================================
check_deps

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GAS WIT Example${NC}"
echo -e "${GREEN}========================================${NC}"
echo "  Project dir : $PROJECT_DIR"
echo "  WIT dir     : $WIT_DIR"
echo "  Temp dir    : $TEMP_DIR"
echo "  Dim=$DIM, Base=$MAX_ELEMENTS, Query=$MAX_QUERIES, K=$K"
echo "  Parts       : $NUM_PARTS (~117k rows)"
echo -e "${GREEN}========================================${NC}"

# ---- Step 1: Download ----
if all_parquets_exist; then
    echo -e "\n${YELLOW}[Step 1/4] Parquets ready, skipping download${NC}"
else
    echo -e "\n${YELLOW}[Step 1/4] Downloading WIT parquets (first ${NUM_PARTS} parts)...${NC}"
    mkdir -p "$WIT_DIR"
    for i in $(seq 0 $((NUM_PARTS - 1))); do
        fname=$(printf "$WIT_PATTERN" "$i")
        dst="$WIT_DIR/$fname"
        if [[ -f "$dst" ]]; then
            echo "  [skip] $fname exists"
            continue
        fi
        url="$HF_ENDPOINT/datasets/$WIT_DATASET/resolve/main/data/$fname?download=true"
        echo "  [dl] $fname ..."
        wget -q --show-progress -O "$dst" "$url"
    done
    echo -e "${GREEN}[Step 1/4] Download done${NC}"
fi

# ---- Step 2: Extract ----
if extract_outputs_exist; then
    echo -e "\n${YELLOW}[Step 2/4] fvecs/bmeta/qmeta ready, skipping extraction${NC}"
else
    echo -e "\n${YELLOW}[Step 2/4] Extracting vectors + bmeta + qmeta from parquets...${NC}"
    mkdir -p "$TEMP_DIR" "$CACHE_DIR"
    python3 "$SCRIPTS_DIR/extract_wit.py" "$WIT_DIR" "$TEMP_DIR"
    echo -e "${GREEN}[Step 2/4] Extraction done${NC}"
fi

# ---- Step 3: Build ----
echo -e "\n${YELLOW}[Step 3/4] Building project...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$PROJECT_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j"$(nproc)" bench gengt > /dev/null 2>&1
cd "$PROJECT_DIR"
echo -e "${GREEN}[Step 3/4] Build done${NC}"

# ---- Step 4: Ground truth ----
GT_FILE=$(ls "$CACHE_DIR"/gt_*_k"$K".ivecs 2>/dev/null | head -1 || true)
if [[ -n "$GT_FILE" && -f "$GT_FILE" ]]; then
    echo -e "\n${YELLOW}[Step 4/4] Ground truth found ($(basename "$GT_FILE")), skipping${NC}"
else
    echo -e "\n${YELLOW}[Step 4/4] Generating ground truth...${NC}"
    mkdir -p "$CACHE_DIR"
    "$BUILD_DIR/gengt" \
        "$DIM" "$MAX_ELEMENTS" "$MAX_QUERIES" "$K" \
        "$CACHE_DIR" \
        "$TEMP_DIR/$BASE_NAME" \
        "$TEMP_DIR/$BMETA_NAME" \
        "$TEMP_DIR/$QUERY_NAME" \
        "$TEMP_DIR/$QMETA_NAME" > /dev/null 2>&1
    echo -e "${GREEN}[Step 4/4] Ground truth done${NC}"
fi

# ---- Benchmark (always run) ----
echo -e "\n${YELLOW}[Benchmark] Running benchmark...${NC}"
"$BUILD_DIR/bench" \
    "$DIM" "$MAX_ELEMENTS" "$MAX_QUERIES" "$K" \
    "$CACHE_DIR" \
    "$TEMP_DIR/$BASE_NAME" \
    "$TEMP_DIR/$BMETA_NAME" \
    "$TEMP_DIR/$QUERY_NAME" \
    "$TEMP_DIR/$QMETA_NAME" \
    "$ONLY_RUN_IDX" "$REPEAT" "$N_SEG" "$BATCH_SIZE" \
    "$QUERY_SEQ_MODE" \
    "${EFS[@]}"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  WIT Example finished${NC}"
echo -e "${GREEN}========================================${NC}"
