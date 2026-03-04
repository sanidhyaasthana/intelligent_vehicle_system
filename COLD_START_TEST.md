# COLD START TEST INSTRUCTIONS

To verify full reproducibility and artifact regeneration:

1. Delete all contents of the following directories:
   - results/
   - cache/ (if present)

2. Run the full evaluation pipeline using the provided config and scripts:
   - python evaluation/run_full_fusion.py --config configs/clean_base.yaml

3. Confirm that the following artifacts are regenerated in results/<timestamp>/:
   - config.yaml
   - metrics.json
   - eer.txt
   - predictions.parquet
   - roc_curve.png
   - det_curve.png
   - latency.json
   - fusion_weights.json
   - manifest.json

4. Check that predictions.parquet contains all required columns and that all integrity checks pass (no silent failures).

5. If all artifacts are present and valid, the system passes the reproducibility gate.

# End of COLD START TEST
