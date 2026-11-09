# Released model weights

The public submission image contains the five checkpoints used by `config/routing.csv`. The image is available anonymously at this immutable reference:

```text
ghcr.io/alstar8/b1k-evo-submit@sha256:1ab7d94f5b3ecc347dbb9adccc2d9212cfbd2ed4aad078bda82faad607a240c5
```

The Linux/amd64 manifest has 64 layers, approximately 71.1 GiB compressed. The image configuration records checkpoint copies under `/opt/checkpoints/`. This distribution supplies the exact packaged inference parameters and normalization assets alongside the inference runtime.

| Source route | Identity recorded by the source | Restore to |
|---|---|---|
| `comet2` | group2_hour_a01, step 2498 | `checkpoints/comet2/` |
| `comet2_1` | group1_7h_a01, step 20408 | `checkpoints/comet2_1/` |
| `comet3` | all63_1h_a01, step 2540 | `checkpoints/comet3/` |
| `larchenko_ckpt2` | Larchenko 2025 checkpoint_2 | `checkpoints/larchenko/checkpoint_2/` |
| `larchenko_ckpt3` | Larchenko 2025 checkpoint_3 | `checkpoints/larchenko/checkpoint_3/` |

The checkpoint names and steps identify artifacts; they do not establish a complete training recipe. COMET checkpoint roots must contain `params/` and `assets/behavior-1k/2026-challenge-rgb224/norm_stats.json`.

## Extract for a source build

From a clean checkout, with Docker installed and sufficient disk space:

```bash
python3 scripts/extract_release_checkpoints.py
python3 server/routing.py config/routing.csv
python3 scripts/check_comet2_checkpoint.py checkpoints/comet2
python3 scripts/build_docker.py --tag vera-b1k:source
```

The extractor pulls the pinned image, creates a stopped container, reads its routing table and copies the five distinct checkpoint roots into the source layout. It never starts the container or loads a model, so extraction does not need a GPU. It stages the copies, verifies their `params/` and `assets/` directories, checks `comet2` against the bundled manifest, and removes the temporary container. Existing checkpoint destinations cause an error before any pull.

Checkpoints are excluded from Git and the main Docker context. `build_docker.py` uses named contexts and copies each distinct checkpoint once. Allow space for the downloaded image, its unpacked layers, extracted checkpoints and any subsequent build.

## Available checksums

`solutions/comet2/checkpoint_provenance.json` records all inference files for `comet2`. Its canonical fingerprint is:

```text
9afdb712de022c058fc10495f8aca921c24b38291dd40aca9950c38805799923
```

The verifier hashes all listed files, validates sizes, rejects missing or extra files in `params/` and `assets/`, and checks the canonical aggregate. The release does not include equivalent independent file manifests for `comet2_1` or `comet3`; their distribution is pinned by the OCI image digest. Do not use the group2 verifier to claim verification of another checkpoint.

## Upstream Larchenko distribution

The upstream model repository offers an alternative source for the adapted 2025 checkpoints:

```bash
hf download IliaLarchenko/behavior_submission \
  --include 'checkpoint_2/*' 'checkpoint_3/*' \
  --local-dir checkpoints/larchenko
```

For the exact released package, use extraction from the image. The command above follows the upstream default revision and is not an independent pin to the files in the image. Preserve upstream model and tokenizer licenses.

## Public artifact scope

As checked on November 9, 2026, [VERA-VLA](https://huggingface.co/VERA-VLA) publicly lists the [validation dataset](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission), with JSON outputs and videos. It does not list separate public COMET model repositories. The weights are distributed through the public submission image referenced by the dataset's submission README.

Release preparation verified anonymous registry access, the image digest and checkpoint layer metadata. It did not download the entire image, extract its full weight payload, rebuild it or run GPU inference. Reproducing those operations requires the resources described in the reproduction guide.
