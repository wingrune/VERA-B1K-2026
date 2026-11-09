Model checkpoints live here. The folder is in git, its contents are not (gitignored); `scripts/build_docker.py` copies the
checkpoints the routing table names into the image.

Layout the routing template expects:

    checkpoints/larchenko/checkpoint_2/{params,assets,...}
    checkpoints/larchenko/checkpoint_3/...
    checkpoints/larchenko/checkpoint_4/...

Download (Larchenko 2025 submission, Hugging Face):

    hf download IliaLarchenko/behavior_submission --include "checkpoint_2/*" "checkpoint_3/*" "checkpoint_4/*" --local-dir checkpoints/larchenko

Other models: put them under checkpoints/<model>/ and write that relative path into config/routing.csv.
