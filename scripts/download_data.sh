#!/usr/bin/env bash
# Download training data.
#
#   bash scripts/download_data.sh pcam                 # PatchCamelyon (~7.5 GB compressed) from Zenodo
#   bash scripts/download_data.sh camelyon             # a few CAMELYON16 slides + XML annotations from AWS
#   N_TUMOR=8 N_NORMAL=3 DATA=/tmp/data bash scripts/download_data.sh camelyon
#
# CAMELYON is on the AWS Registry of Open Data (s3://camelyon-dataset, no account needed).
# Each slide is 0.5-3 GB, so start with a handful.
set -euo pipefail
DATA=${DATA:-data}
N_TUMOR=${N_TUMOR:-6}
N_NORMAL=${N_NORMAL:-2}

pcam() {
  mkdir -p "$DATA/pcam"
  for split in train valid test; do
    for t in x y; do
      f="camelyonpatch_level_2_split_${split}_${t}.h5"
      if [ ! -f "$DATA/pcam/$f" ]; then
        echo "downloading $f"
        wget -q --show-progress -O "$DATA/pcam/$f.gz" "https://zenodo.org/records/2546921/files/$f.gz?download=1"
        gunzip -f "$DATA/pcam/$f.gz"
      fi
    done
  done
  ls -lh "$DATA/pcam"
}

camelyon() {
  command -v aws >/dev/null || pip install -q awscli
  mkdir -p "$DATA/camelyon16/images" "$DATA/camelyon16/annotations"
  listing="$DATA/camelyon16/s3_listing.txt"
  # list once; the bucket layout is discovered rather than hard-coded
  [ -s "$listing" ] || aws s3 ls --no-sign-request --recursive s3://camelyon-dataset/CAMELYON16/ > "$listing"
  pick() { grep -E "$1" "$listing" | awk '{print $4}' | sort | head -n "$2"; }
  for key in $(pick '/tumor_[0-9]{3}\.tif$' "$N_TUMOR") $(pick '/normal_[0-9]{3}\.tif$' "$N_NORMAL"); do
    name=$(basename "$key")
    [ -f "$DATA/camelyon16/images/$name" ] || aws s3 cp --no-sign-request "s3://camelyon-dataset/$key" "$DATA/camelyon16/images/$name"
    sid="${name%.tif}"
    xml_key=$(grep -E "/${sid}\.xml$" "$listing" | awk '{print $4}' | head -n 1 || true)
    if [ -n "$xml_key" ] && [ ! -f "$DATA/camelyon16/annotations/$sid.xml" ]; then
      aws s3 cp --no-sign-request "s3://camelyon-dataset/$xml_key" "$DATA/camelyon16/annotations/$sid.xml"
    fi
  done
  ls -lh "$DATA/camelyon16/images" "$DATA/camelyon16/annotations"
}

case "${1:-}" in
  pcam) pcam ;;
  camelyon) camelyon ;;
  *) echo "usage: $0 {pcam|camelyon}"; exit 1 ;;
esac
