# Running the pipeline on Kaggle

Local runs crashed the machine: 8 GB of RAM could not hold two countries'
TF-IDF matrices plus their feature matrices at once, and the resulting swap
thrash triggered a watchdog reset. Kaggle gives ~30 GB of RAM, which removes
that failure mode. It has fewer cores, so it is slower but it finishes.

AWS was ruled out: the account is on the **Free Plan**, whose largest permitted
instance is `m7i-flex.large` (2 vCPU / 8 GiB) -- smaller than the laptop.

---

## One-time: upload the dataset

Kaggle datasets are capped well above what we need, and the data never changes,
so upload it once and reuse it for every run.

1. kaggle.com → **Datasets** → **New Dataset**
2. Upload the `student_resource/dataset/` folder, preserving the layout:

   ```
   train/train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
   test/test_source1.tsv,  test_source2.tsv,  test_source3.tsv
   ```

3. Title it something like `amazonml-er-data`. Keep it **Private**.
4. Wait for it to finish processing (~2.4 GB).

Faster alternative via the Kaggle CLI:

```bash
pip install kaggle          # put your token in ~/.kaggle/kaggle.json
cd student_resource
kaggle datasets init -p dataset
# edit dataset/dataset-metadata.json -> set "title" and "id"
kaggle datasets create -p dataset --dir-mode zip
```

---

## Notebook settings (all four matter)

| Setting | Value | Why |
|---|---|---|
| Internet | **ON** | `pip install` and `git clone` |
| Accelerator | **None** | This is CPU work. A GPU session gives *fewer* vCPUs |
| Persistence | Files only | Keeps `/kaggle/working` between runs |
| Dataset | attach `amazonml-er-data` | Mounts it read-only under `/kaggle/input/` |

Run each stage with **Save Version → Save & Run All (Commit)**. That executes
headless for up to 12 hours and persists the output, so an idle browser tab
cannot kill the job.

---

## Stages

Paste `kaggle_run.py` into a single cell and change `STAGE` between runs.
Countries are independent partitions, so they can run separately and be merged.

| # | `STAGE` | Also attach | Est. time | Produces |
|---|---|---|---|---|
| 1 | `"train"` | — | ~1.5 h | `artifacts/model.txt`, `threshold.json` |
| 2 | `"India"` | output of #1 | ~4 h | `output/*_India.tsv` |
| 3 | `"US"` | outputs of #1, #2 | ~2.5 h | `output/*_US.tsv` |
| 4 | `"France"` | outputs of #1–#3 | ~0.5 h | `output/*_France.tsv` |
| 5 | `"merge"` | outputs of #1–#4 | ~2 min | final `matching_results.tsv`, `candidate_pairs.tsv` (+ `.gz`) |

To chain them: after each run finishes, open the next notebook and add the
previous run's **output** as an input dataset. `kaggle_run.py` detects prior
artifacts and partial outputs automatically and copies them forward.

Running all four predict stages in one go is possible (set `STAGE` to each in
sequence in separate cells) but risks the 12 h ceiling with no margin; a failure
at hour 8 loses everything, whereas a per-country failure loses one partition.

---

## After the merge

1. Download `output/matching_results.tsv.gz` and `output/candidate_pairs.tsv.gz`
2. Decompress locally into `Lassi_submission/output/`
3. Validate before submitting, from `student_resource/`:

   ```bash
   python3 utils/validate_submission.py \
       --matching ../Lassi_submission/output/matching_results.tsv \
       --candidate ../Lassi_submission/output/candidate_pairs.tsv \
       --test-dir dataset/test
   ```

4. Upload `matching_results.tsv` to the Portal.

The merge step already verifies that every test Source-1 entity appears exactly
once across the partials, so a missing country fails there rather than at
submission.
