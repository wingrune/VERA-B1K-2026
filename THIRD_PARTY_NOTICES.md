# Third-party acknowledgments

This repository preserves the source snapshot's Apache-2.0 root license and bundled third-party notices. Attribution and component licenses continue to apply to redistributed files.

| Component | Use | Preserved notices or source |
|---|---|---|
| Ilia Larchenko et al., BEHAVIOR 2025 solution | VLA backend and 2026 adapter | [Vendored license](solutions/larchenko_2025/upstream/LICENSE), [upstream repository](https://github.com/IliaLarchenko/behavior-1k-solution), [report](https://arxiv.org/abs/2512.06951) |
| OpenPI / Physical Intelligence | π0.5 inference and action processing | [Vendored license](solutions/larchenko_2025/upstream/openpi/LICENSE), [upstream](https://github.com/Physical-Intelligence/openpi) |
| COMET runtime components | Bundled OpenPI, LeRobot, Gemma-related runtime and tokenizer | [Bundled notices](solutions/comet2/licenses/) |
| b1k_rl_guys controllers | Task-specific classical policies and static priors | Upstream identities and revisions recorded in `tasks/*/solution.json`; [source](https://github.com/tttonyalpha/b1k_rl_guys) |
| BIMBO_EvoX and B1K_hacker planners | Task-specific planning programs and evolution artifacts | Task README files and `solution.json` record origin, commits or artifact hashes |
| cuRobo / NVIDIA | GPU inverse kinematics and planning dependency | [Upstream](https://github.com/NVlabs/curobo), revision pinned in `docker/Dockerfile` |
| BEHAVIOR-1K / Stanford | Evaluator, task identifiers, wrapper/config copies and simulator | [Upstream](https://github.com/StanfordVL/BEHAVIOR-1K) |

The root Apache-2.0 license covers the project code as inherited from the source release; it does not override component licenses. Checkpoints and tokenizer/model assets retain applicable upstream terms, including the bundled Gemma notice. The validation dataset declares `license: other`; simulator assets require the upstream installation and terms. No simulator asset bundle is redistributed here.

The blog's radio poster is a frame from the VERA team's public episode `turning_on_radio_301_0.mp4`, pinned to the dataset revision in `docs/data/evidence-manifest.json`. Embedded videos stream from that same public dataset.
