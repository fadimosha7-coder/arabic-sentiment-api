# Data card

## ArSenTD-LEV (training, validation, test)

- 4,000 rows from Levantine Arabic tweets (1,000 each: Jordan, Lebanon, Palestine, Syria). Target-based annotation: one row per (tweet, target), 5 sentiment levels. 19 topics; 5 topics make 96.6% of the rows.
- Source of my copy: Google Drive (Day 5). License: No known license (see LICENSES.md).
- My processing: 2,889 unique tweets; aggregated per tweet: 1,291 negative, 889 positive, 590 neutral-only (dropped), 39 mixed (dropped). Near-identical tweets grouped; 1 conflicting group removed. Final 2,178: 1,523 train / 326 val / 329 test (stratified, by tweet group, seed 42).
- Known issues: 30% of rows repeat an earlier tweet (multiple targets); the Country column is probably where the tweet was collected, not its dialect (an Egyptian tweet labelled Palestine in my 30-tweet read); strong topic skew; about 22% of unique tweets dropped as neutral or mixed; no author id, so same-author leakage cannot be excluded.
- Privacy: tweets may contain usernames and links. Model input removes @mentions and URLs. Raw data is not published.

## AJGT (external test only)

- 1,800 tweets, balanced 900/900 (1 = positive), no exact-text overlap with ArSenTD-LEV after normalization. License: No explicit license (see LICENSES.md).
- Used once, never for training or model selection.
