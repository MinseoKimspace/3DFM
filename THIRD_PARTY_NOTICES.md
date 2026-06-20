# Third-Party Notices

This project does not vendor the LION metric implementation.

The shape evaluation backend is imported at runtime from an external LION
checkout when running `3dIMF/src/eval.py`.

- LION repository: https://github.com/nv-tlabs/LION
- OGPP reference for using LION CD/EMD metrics:
  https://github.com/swang3081/OGPP/blob/main/BASELINES.md

LION's `utils/evaluation_metrics_fast.py` and its CD/EMD extensions are not
copied into this repository. To use them, clone/build LION separately and pass
its root with `--lion-root`, or add it to `PYTHONPATH`.
