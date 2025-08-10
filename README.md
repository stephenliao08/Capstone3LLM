Capstone3LLM" 
ACME DIALOGUE SUMMARIZATION (SAMSUN)

BLUF
Build and compare BART, BERT2BERT, and T5 on the SAMSum dataset to generate concise chat recaps. Ship the model that reaches ROUGE-1/ROUGE-L ≈ 0.4–0.5 with sub-second inference to reduce catch-up time in noisy group chats.

OVERVIEW
Acme’s group chats are overloaded—important updates get buried and catching up takes 5–10 minutes. This project delivers a proof-of-concept AI summarizer trained on SAMSum (~16k chats) to produce short, messenger-style recaps. The pipeline covers:

Data loading, cleaning, tokenization

Model training (BART, BERT2BERT, T5)

Evaluation (ROUGE + spot-checks)

Inference utilities
Target: ROUGE-1/ROUGE-L around 0.4–0.5 and sub-second latency.

DATASET

Name: knkarthick/samsum (Hugging Face Datasets)

Size: ~16,000 conversation–summary pairs

Splits: train ~14.7k, val ~818, test ~819

Style: informal, multi-turn, messenger-like

ENVIRONMENT / INSTALLATION
Requirements

Python 3.9+

PyTorch (GPU recommended)

transformers, datasets, evaluate, nltk, matplotlib, seaborn (optional), wordcloud

Install (examples)

pip install "transformers>=4.36" datasets evaluate nltk matplotlib seaborn wordcloud

Optional environment flags:

USE_TF=0

TOKENIZERS_PARALLELISM=false

TORCH_COMPILE_DISABLE=1

NLTK

In code: nltk.download("punkt", quiet=True)

SUGGESTED PROJECT STRUCTURE
.

notebooks/

samsum_summarization.ipynb

models/

bart-samsum/

t5-samsum/

bert2bert-samsum/

src/

data.py (load/clean/tokenize)

train_bart.py

train_t5.py

train_bert2bert.py

eval.py (ROUGE + plots)

infer.py (batch inference)

README.txt

DATA CLEANING (LIGHT)

Unicode normalize (NFKC)

Collapse excessive spaces; preserve line breaks for dialogue; strip control chars

Optional: filter very short samples

MODELS

BART (pretrained seq2seq baseline)

Tokenize dialogue as encoder input (max length ~512)

Tokenize summary as decoder labels (max length ~128)

Use DataCollatorForSeq2Seq for padding and -100 label handling

Train 3+ epochs with predict_with_generate=True and generation settings (beams ~4, max length ~128)

Expect strong baseline ROUGE

T5 / FLAN-T5

Always pair T5 model with T5 tokenizer

Prepend "summarize: " to inputs

Similar training setup to BART; learning rate often a bit higher (e.g., 1e-4)

With proper prefix + training, quality improves significantly

BERT2BERT (EncoderDecoderModel)

Build encoder–decoder from bert-base-uncased

Set decoder_start_token_id=CLS, eos=SEP, pad=PAD; no_repeat_ngram_size=3

Expect lower ROUGE without longer training (not pretrained for generation)

EVALUATION

Metric: ROUGE (evaluate.load("rouge"))

Compute with decoded text (remove -100, skip special tokens, light whitespace normalization)

Report as percentages (0–100)

Also perform spot-check inference: print dialogue, reference, prediction

PLOTTING ROUGE

Simple bar chart for ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum from the evaluation dict

If keys include eval_rouge*, convert fractions (≤1) to percentages

INFERENCE (SPOT-CHECK)

Batch tokenize dialogues

Generate with beams (2–6), no_repeat_ngram_size=3, length_penalty≈1.0–1.2

For T5, remember "summarize: " prefix

Print side-by-side with references for quick sanity checks

RESULTS (EXAMPLE PATTERN; YOURS MAY DIFFER)

BART-base: ROUGE-1 ≈ 45–50, ROUGE-L ≈ 40–47

FLAN-T5: lower without enough tuning; improves with proper prefix + longer training

BERT2BERT: low without long training, as it lacks seq2seq pretraining
Target remains ROUGE-1/ROUGE-L ≈ 0.4–0.5 and sub-second inference.

DEPLOYMENT TIPS (SUB-SECOND)

Use fewer beams (2–4), cap max_new_tokens (~64–128)

Batch requests; pre-warm model; keep it resident

Consider 8-bit/4-bit quantization

Use pinned memory and persistent workers

Cache frequent prompts if applicable

TROUBLESHOOTING

Gibberish outputs: model/tokenizer mismatch (e.g., T5 model with BART tokenizer). Always pair correctly; for T5, use the "summarize: " prefix.

Transformers version differences: argument names can change (evaluation_strategy vs eval_strategy). Align save/eval strategies if using load_best_model_at_end.

Very low BERT2BERT ROUGE: expected without long training; use BART/T5 for MVP.

OOM: reduce batch size/sequence lengths; use gradient accumulation or gradient checkpointing.

DELIVERABLES

Jupyter notebook with full pipeline

Evaluation metrics (ROUGE) and sample outputs

Plots (ROUGE bar charts, basic EDA: word counts, wordcloud)

3–5 page pitch: problem, approach, timeline, risks, results

ACKNOWLEDGEMENTS

SAMSum dataset

Hugging Face: transformers, datasets, evaluate

CONTACT / NEXT STEPS

Start with BART to establish a strong, shippable baseline

Train T5 longer with correct tokenizer/prefix to close the gap

Optimize inference for sub-second latency before user testing

