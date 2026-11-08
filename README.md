# Strata engineering notes

Read [What makes a local agent fast enough to use?](https://macoredroid.github.io/strata-local-inference/), a Strata field note on local Qwen serving, measurement, and automatic media switching.

On a Linux GPU host with Docker and NVIDIA container access, run `bash recipes/serve.sh 5090` or `bash recipes/serve.sh spark`. The script stages the pinned checkpoints, verifies their hashes, and starts the selected profile on loopback. The article explains hardware requirements and how to check the result.

Edit `article.md` and run `python3 build.py` with Pandoc installed to regenerate `index.html`. GitHub Pages serves the committed HTML and stylesheet directly from `main`.
