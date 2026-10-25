# RedSpell

Grammar and spelling correction for English text (Projet 15). A T5 model is fine-tuned on real learner data (Lang-8) and served through a small Streamlit app that underlines errors, proposes corrections, and lets you accept or reject each one.

## Results

Test set: the official `entries.test` split of the Lang-8 English corpus v1.0 (human corrections). GLEU is the grammatical-error-correction variant (Napoles et al., 2016).

| Model | Test GLEU | Exact match % | Sentences changed % |
|---|---|---|---|
| Copy the input (floor) | 35.43 | 2.04 | 0.00 |
| SymSpell (spelling only) | 37.46 | 3.53 | 18.51 |
| `t5-small` (3 epochs) | 47.81 | 12.47 | 65.02 |
| **`t5-base` (2 epochs)** | **50.95** | **15.79** | 70.87 |

The model was chosen on validation GLEU (1,000 validation sentences, same for both models): `t5-base` 52.26 vs `t5-small` 48.27 at the end of training. The test set was not used for any choice.

### Error analysis

Sample of 296 test sentences: 200 whose human reference differs from the input, and the 96 sentences whose reference leaves the input unchanged (all that exist in the test split). Edits are extracted by word-level alignment and typed with simple rules, so the categories are approximate. "Recovered" means the model made exactly the human edit; a different but valid correction counts as a miss.

| Edit type | Human edits | Recovered | Recall |
|---|---|---|---|
| phrase rewrite | 83 | 8 | 9.6% |
| word choice | 66 | 8 | 12.1% |
| missing word | 59 | 12 | 20.3% |
| word form (tense, number) | 56 | 30 | 53.6% |
| extra word | 32 | 7 | 21.9% |
| article | 31 | 21 | 67.7% |
| preposition | 24 | 5 | 20.8% |
| punctuation | 16 | 1 | 6.2% |
| spelling | 13 | 7 | 53.8% |
| agreement | 13 | 6 | 46.2% |
| casing | 11 | 7 | 63.6% |

Spelling errors are only 13 of 404 human edits (3.2%): Lang-8 contains few typos, so isolated misspellings are a weak point. The app therefore flags words that are not in a dictionary and offers suggestions, never applying them automatically.

**Over-correction.** The model changed 20 of the 96 already-correct sentences (20.8%). Manual review of all 20 (one annotator): 16 improvements (errors the reference missed), 3 acceptable alternatives, 1 harmful (an unbalanced quotation mark). Before a fix for apostrophe noise in the data (backticks and spaced contractions, repaired at inference by `normalize` and `clean_output` in `src/model/infer.py`), 3 were harmful.

### Speed (CPU only, no GPU)

Five-sentence paragraph, `t5-base`, median of 3 runs after warm-up: beam 4 = 2.7 s, beam 2 = 2.1 s (same output on this paragraph). A single 9-word sentence takes about 1.3 s. The first request after starting the app also loads the model (tens of seconds). int8 quantization was slower (3.1 s) and less accurate on the test machine, so it is off by default.

## Repository layout

```
app/app.py                       Streamlit interface
src/data/build_pairs.py          parse, clean and split the Lang-8 pairs
src/embeddings/tokenizer_demo.py subword tokenization and embeddings demo
src/model/infer.py               model loading and text correction
src/utils/diff.py                word-level diff used for the highlighting
src/evaluation/metrics.py        GLEU and sentence-level rates
src/evaluation/error_analysis.py error analysis by edit type
notebooks/                       training notebook (Kaggle, GPU)
reports/                         error_summary.csv, error_over_correction.csv
data/, models/                   local only, not in git
```

## Setup

```
pip install -r requirements.txt
```

## Usage

1. **Build the data** (needs the Lang-8 English corpus v1.0, see below):
   `python -m src.data.build_pairs --src path/to/lang-8-en-1.0 --out data`
2. **Train**: run `notebooks/01_data_and_training_kaggle.ipynb` on a GPU (about 1.5 h for `t5-base`). Put the resulting model folder in `models/t5-base`.
3. **App**: `streamlit run app/app.py`
4. **Error analysis**: `python -m src.evaluation.error_analysis --model models/t5-base --n 200 --identity 100`
5. **Tokenizer demo**: `python -m src.embeddings.tokenizer_demo`

Model weights are not included in the repository. They are re-created by the notebook.

## Data

Lang-8 Learner Corpus of English v1.0 (Mizumoto et al., 2011; Tajiri et al., 2012): 494,248 corrected training sentences and 4,769 test sentences after parsing; the training set is capped at 100,000 pairs after cleaning (detokenization, ASCII filter, length filter, de-duplication, removal of any training sentence that also appears in the test split).
After cleaning: 100,000 training pairs, 3,000 validation pairs and 4,700 test pairs.

The corpus is available for research and education only and may not be redistributed, so **it is not included**. Request it from the authors (NAIST Lang-8 Learner Corpora page). Predictions and notebook outputs that contain corpus text are also kept out of the repository.

## Limitations

- Isolated spelling mistakes are often missed (see above); misspellings get suggestions, not automatic fixes.
- Lang-8 references are single human corrections of uneven quality and consistency (casing, apostrophes, quotes), so exact match and GLEU understate the model's quality, and a lowercase `i` is often left as is.
- Quotation marks can come out unbalanced.
- Homophones such as *their*/*there* can be missed.
- The over-correction review is small (96 sentences, 20 judged, one annotator).
- Sentence splitting is rule-based and can split at abbreviations. The model is English only and trained on sentences, not documents.
- About 1.1% of training sentences contain characters outside the T5 vocabulary and were dropped.
- Retraining with better cleaning of apostrophes and quotes, and with synthetic typos added, is future work.

## References

- Mizumoto, T., Komachi, M., Nagata, M., Matsumoto, Y. (2011). Mining Revision Log of Language Learning SNS for Automated Japanese Error Correction of Second Language Learners. *IJCNLP*, pp. 147-155.
- Tajiri, T., Komachi, M., Matsumoto, Y. (2012). Tense and Aspect Error Correction for ESL Learners Using Global Context. *ACL (Short Papers)*, pp. 198-202.
- Napoles, C., Sakaguchi, K., Tetreault, J. (2016). GLEU Without Tuning. arXiv:1605.02592.
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. *JMLR* 21(140).
- Garbe, W. SymSpell, https://github.com/wolfgarbe/SymSpell (used through the `symspellpy` package).