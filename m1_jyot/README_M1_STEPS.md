# M1 — Data Engineering & Cleaning: what to actually do

Scripts in this folder implement U1.1–U1.7 from the proposal (slides 5–8) and
have been tested end-to-end on a synthetic stand-in file with the same shape
as the real dataset (5,110 rows, 12 columns, ~4.9% positive, 201 missing BMI).
You still need to run them on the real CSV.

## Steps, in order

1. **Download the real dataset** from Kaggle (`fedesoriano/stroke-prediction-dataset`).
   Save it as `healthcare-dataset-stroke-data.csv` in this folder.

2. **Record checksum + licence** — run:
   ```
   python preprocess.py healthcare-dataset-stroke-data.csv
   ```
   This prints a SHA-256 checksum and writes `stroke_typed.parquet`. Copy the
   checksum into `LICENCE_NOTE.md` (template below) along with the licence
   terms from the Kaggle dataset page. This satisfies U1.1 and the
   "licence or provenance unclear" risk on slide 8.

3. **Check the printed flag count** against slide 5's provisional facts
   (5,110 rows, 249 positive, 201 missing BMI, gender has one "Other").
   If your real download doesn't match these numbers, the dataset may have
   changed since the proposal was written — flag this to the team before
   going further, since M3–M5's numbers all assume these facts.

4. **Freeze the splits**:
   ```
   python make_splits.py stroke_typed.parquet
   ```
   Writes `splits.json`. Commit this file and never regenerate it. Send it
   to Chirantan (M2), Arth (M3), Dhruv (M4) — everyone must fit on the same
   folds or comparisons across models aren't valid.

5. **Generate the data-quality report**:
   ```
   python data_quality_report.py healthcare-dataset-stroke-data.csv
   ```
   Writes `data_quality_report.md`. Read through the flagged-row table —
   this is where you make (and document) judgment calls like what to do
   about the `gender == "Other"` row or any `work_type`/`age` mismatches.
   Don't drop rows to make the numbers look cleaner; log the decision instead.

6. **Hand off to Chirantan (M2)**: `stroke_typed.parquet` (via preprocess.py's
   `load_and_type`), `schema.yaml`, and `CleanTransformer` from `preprocess.py`.
   He fits your transformer as step 1 of his own pipeline, inside training
   folds only — that's already wired into how `CleanTransformer.fit/transform`
   is split.

7. **Before you call this done**, re-run steps 2–5 from a clean clone and
   confirm you get byte-identical `stroke_typed.parquet` and the same
   `splits.json` — that's your acceptance criterion on slide 8
   ("Two runs produce byte-identical outputs").

## Where "real-world accuracy" fits into your job specifically

Your slice can't fix generalization by itself — that's mostly M3/M4's
modeling and M5's honesty about limitations. But three things at your stage
materially affect it:

- **Don't over-clean.** Every anomaly you flag-and-keep (the "Other" gender
  row, out-of-range glucose/BMI values, the children-with-adult-age rows)
  is a row a real clinical dataset would also have. If you quietly drop or
  "fix" these to make U1.7's report look tidier, the model trains on an
  easier distribution than production data and the accuracy gap on real
  data gets worse, not better.
- **Missing BMI is a signal, not noise.** Keep `bmi_missing` as a feature
  (already done in `CleanTransformer`) rather than just imputing and
  discarding the fact it was missing — missingness patterns often carry
  real information in clinical data.
- **Say this out loud to the team now, not at the end:** the whole
  pipeline is being validated against one held-out slice of the *same*
  Kaggle dataset. That tells you if the model generalizes within this
  dataset's distribution — it says nothing about a different hospital's
  intake form, different regional glucose/BMI norms, or different
  smoking-status coding. If the professor is going to test on real-world
  data, flag to the group now that Part 3 should include a second,
  independently-sourced public health dataset (even a small one, e.g. a
  regional or CDC-style survey with overlapping fields) purely as an
  external sanity check — not for training, just to see if performance
  collapses outside this dataset's quirks. That's a scope decision for
  the whole team (it touches M3's evaluation protocol and M5's model
  card), worth raising in your weekly integration review since you're
  leading it.
