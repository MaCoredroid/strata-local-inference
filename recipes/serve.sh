#!/usr/bin/env bash
# Download and verify the selected stock profile, then serve on localhost.
set -euo pipefail

profile=${1:-}
if [[ "$profile" != 5090 && "$profile" != spark ]]; then
  printf 'Usage: %s {5090|spark} [--download-only]\n' "$0" >&2
  exit 2
fi
action=${2:-serve}
if [[ "$action" != serve && "$action" != --download-only ]]; then
  printf 'Unknown option: %s\n' "$action" >&2
  exit 2
fi

asset_dir=${STRATA_MODEL_DIR:-"$PWD/models-$profile"}
mkdir -p "$asset_dir"
asset_dir=$(cd "$asset_dir" && pwd -P)
cd "$asset_dir"

verify() {
  printf '%s  %s\n' "$1" "$2" | sha256sum --check --status
}
fetch() {
  local repo=$1 revision=$2 remote_file=$3 local_file=$4 digest=$5
  if [[ -f "$local_file" ]]; then
    if ! verify "$digest" "$local_file"; then
      printf 'Checksum mismatch: %s. Move the existing file aside and retry.\n' "$local_file" >&2
      return 1
    fi
    return
  fi
  local partial="$local_file.partial"
  if [[ ! -f "$partial" ]] || ! verify "$digest" "$partial"; then
    curl --fail --location --retry 3 --continue-at - --output "$partial" \
      "https://huggingface.co/$repo/resolve/$revision/$remote_file"
  fi
  if ! verify "$digest" "$partial"; then
    printf 'Checksum mismatch: %s. Inspect or remove the partial file before retrying.\n' "$partial" >&2
    return 1
  fi
  mv "$partial" "$local_file"
}

if [[ "$profile" == 5090 ]]; then
  target=huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF
  target_rev=1c69098e9c32db6dc06fc48f1d6cc75ad07da89e
  aux=YTan2000/Qwen3.8-27B-TQ
  aux_rev=4a74bae323adaab36d40d4b8e49b5a1b02a6faf6
  fetch "$target" "$target_rev" Huihui-Qwen3.8-27B-abliterated-UD-DW-Q4_K_M.gguf model.gguf 0c7cfe3060493485bb9a6a51195897b0c1d347a49929206adb9a52170183ea03
  fetch "$target" "$target_rev" mmproj-model-bf16.gguf mmproj.gguf c9a09064683620bea3d3bfed5d4462e1a97a7d2fff7e5045d6862a0a85eeb5b5
  fetch "$aux" "$aux_rev" drafters/Qwen3.8-27B-DFlash2-TQ3.gguf draft.gguf 7f12a657b8c039f6748eb665c650fd0de47284cd8975847fa4e6757cf77a07bd
  fetch "$aux" "$aux_rev" mtp-vocab/atx_65536.txt atx_65536.txt 8405ff0f8970da24b72d68c9e61fa4309fdcf3719ecc1d7626f0ae8bd2a8e326
  args=(--context 204800 --model /fn/model.gguf --name strata-27b --kv q8 -fitl off
        -md /fn/draft.gguf --draft-vocab /fn/atx_65536.txt --draft-max 7
        --reasoning-budget 24576 --reasoning-effort xhigh --mmproj /fn/mmproj.gguf --image-max-tokens 2048)
  gpu=(--gpus all)
  memory=25g
else
  target=huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF
  target_rev=411f1f7349c9711e6f75aa8d20e1612eb3a4d55d
  aux=YTan2000/Qwen3.8-Flash-Next-tqLLM
  aux_rev=d472624dae721fed33a7c720bac866d47bcb70a1
  fetch "$target" "$target_rev" swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf target-1.gguf 9cdb79ad99ef066064628f13b67ace8ef077d02d9d3848a53b11d5c3cbe94332
  fetch "$target" "$target_rev" swift-IQ3_S/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf target-2.gguf 04e1f010edbeacbce3a8948abf0044e858d73a37fc74a0e58abeeff98cbe4f0f
  fetch "$aux" "$aux_rev" mtp-blk48-iq4.gguf mtp-blk48-iq4.gguf 475b777d29216ff33b12275fb6becfa801bc0cb5c146fe7805bbc7e5a8e4d402
  fetch "$aux" "$aux_rev" flashnext-tokenizer.gguf flashnext-tokenizer.gguf e1b687bbf6fdaa4007a2080dc97a1881a7c9bc715630682c5cc6ba1ba409fd68
  fetch "$aux" "$aux_rev" atx_65536.txt atx_65536.txt 8405ff0f8970da24b72d68c9e61fa4309fdcf3719ecc1d7626f0ae8bd2a8e326
  cat > strata-spark.tqfn.tmp <<'MANIFEST'
gguf=target-1.gguf
gguf=target-2.gguf
mtp=mtp-blk48-iq4.gguf
draft_vocab=atx_65536.txt
slots=auto
chunk=8192
layer_slots=314,291,269,160,189,219,248,239,208,135,212,160,227,227,172,128,128,145,227,169,164,197,236,141,152,134,182,137,189,167,167,128,128,135,213,143,153,170,226,128,129,140,161,131,140,140,176,165
MANIFEST
  mv strata-spark.tqfn.tmp strata-spark.tqfn
  args=(--context 262144 --model /fn/strata-spark.tqfn --tokenizer /fn/flashnext-tokenizer.gguf
        --name strata-flash-next --reasoning-budget 16384 --reasoning-effort high)
  gpu=(--device=nvidia.com/gpu=all)
  memory=110g
fi

[[ "$action" == --download-only ]] && exit 0

port=${STRATA_PORT:-8144}
if [[ ! "$port" =~ ^[0-9]+$ ]] || ((10#$port < 1024 || 10#$port > 65535)); then
  printf 'Choose an unprivileged STRATA_PORT between 1024 and 65535.\n' >&2
  exit 2
fi
mounts=(-v "$asset_dir:/fn:ro")
if [[ -n "${STRATA_TQSERVE_BINARY:-}" ]]; then
  if [[ "$profile" != spark || "$STRATA_TQSERVE_BINARY" != /* || ! -x "$STRATA_TQSERVE_BINARY" ]]; then
    printf 'STRATA_TQSERVE_BINARY must be an absolute path to an executable for Spark.\n' >&2
    exit 2
  fi
  mounts+=(-v "$STRATA_TQSERVE_BINARY:/usr/local/bin/tqserve:ro")
fi
exec docker run --rm "${gpu[@]}" --user "$(id -u):$(id -g)" --cap-drop ALL \
  --security-opt no-new-privileges --memory "$memory" --memory-swap "$memory" \
  --shm-size 1g -p "127.0.0.1:$port:8144" "${mounts[@]}" \
  --entrypoint tqserve ghcr.io/turbo-tan/tqllm:1.4.4 \
  --host 0.0.0.0 --port 8144 --np 1 --exact on --reasoning on --prompt-cache on \
  --max-tokens 32768 --reasoning-answer-reserve 0 --temperature 1 --top-p .95 --top-k 20 "${args[@]}"
