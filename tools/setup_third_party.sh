#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"

mkdir -p "${THIRD_PARTY_DIR}"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [ -d "${target_dir}/.git" ] || [ -d "${target_dir}" ]; then
    echo "Skipping ${target_dir}: already exists"
    return 0
  fi

  echo "Cloning ${repo_url} into ${target_dir}"
  git clone "${repo_url}" "${target_dir}"
}

clone_if_missing "https://github.com/tensorflow/tflite-micro.git" \
  "${THIRD_PARTY_DIR}/tflm"
clone_if_missing "https://github.com/riscv-software-src/riscv-isa-sim.git" \
  "${THIRD_PARTY_DIR}/riscv-isa-sim"
clone_if_missing "https://github.com/riscv-software-src/riscv-pk.git" \
  "${THIRD_PARTY_DIR}/riscv-pk"

cat <<'EOF'

Third-party source checkouts are now present under third_party/.

Next steps depend on your machine:
- build and install Spike / pk locally, or
- point the Makefile to existing local installations

This repository intentionally does not commit the large third-party source trees.
EOF
