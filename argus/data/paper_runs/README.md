# Paper-cycle logs

One file per run of the internal paper desk's cycle (`python -m argus.paper.runner --once`, run
on a schedule), named `cycle_<date>_<time>.log`. Each holds the cycle's own JSON report: what it
read, the decisions it wrote to the hash-chained ledger, and anything it refused.

They are evidence, not scratch. Two evaluators read them:

- `eval/cyclecheck.py` checks that every scheduled cycle ran and finished.
- `eval/execution_truth.py` checks every logged decision against the ledger row it claims to have
  written.

The older files are UTF-16, as PowerShell's redirection wrote them; `execution_truth` decodes
either encoding, and `cyclecheck` reads the latest cycle, which is written as UTF-8.
