# Text Summarizer: Qwen2.5 (QLoRA) vs. T5-small

A small benchmark of two fine-tuned models on the same summarisation task, plus a FastAPI service to query them.

The task is **step list → short imperative title**: given the steps of a WikiHow article, the model produces a concise title such as *"Make Banana Pancakes"*. Both models are trained and evaluated on the same data splits, so their scores are directly comparable.

| | Qwen2.5-1.5B-Instruct + QLoRA | T5-small (full fine-tune) |
|---|---|---|
| Type | Decoder-only LLM, 4-bit quantised base with LoRA adapters | Encoder-decoder, all weights trained |
| Notebook | [`notebooks/qwen_text_summarizer.ipynb`](notebooks/qwen_text_summarizer.ipynb) | [`notebooks/t5_small_text_summarizer.ipynb`](notebooks/t5_small_text_summarizer.ipynb) |

## Results

Evaluated on the held-out test split (1,147 examples) with ROUGE (stemming enabled), using the titles as references.

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| T5-small (fine-tuned) | 0.3762 | 0.1388 | 0.3677 | 0.3679 |
| Qwen2.5-1.5B + QLoRA | **0.4510** | **0.2133** | **0.4406** | **0.4409** |

Qwen scores roughly 7 to 7.5 ROUGE points higher on every metric. For Qwen, inference on the full test set took 125.7 s in batches of 16 (about 110 ms per example) on a Colab GPU.

## Data

- Dataset: [`b-mc2/wikihow_lists`](https://huggingface.co/datasets/b-mc2/wikihow_lists) on Hugging Face (11,461 examples, columns `title` and `result`).
- Input: `result` (the step list). Target: `title`.
- Split (shuffled with seed 42): 80% train (9,168), 10% test (1,147), 10% validation (1,146).

## Training setup

**Qwen2.5-1.5B-Instruct (QLoRA)**

- 4-bit NF4 quantisation of the base model; LoRA with `r=16`, `alpha=16`, dropout 0.05 on all attention and MLP projection layers (`q/k/v/o/gate/up/down_proj`).
- Chat-format supervised fine-tuning (`trl` `SFTTrainer`) with loss computed only on the assistant answer.
- 2 epochs, learning rate 2e-4 (cosine schedule), batch size 8 with gradient accumulation 2, max length 512, fp16. Training took about 1 hour 2 minutes (1,146 steps).
- Validation loss was lowest at step 500 (1.308) and rose slightly afterwards (1.322 at the end), so the evaluation uses `checkpoint-500`.

**T5-small**

- Full fine-tuning with the `summarize:` prefix, 5 epochs, learning rate 3e-5, batch size 8 with gradient accumulation 2, weight decay 0.01, 500 warm-up steps, fp16. Training took about 17 minutes (2,865 steps).
- Evaluated with beam search (4 beams, length penalty 0.6, up to 128 new tokens).

## Repository structure

```
.
├── api/
│   ├── app.py                  # FastAPI app factory + uvicorn entry point
│   ├── router.py               # /summarize endpoint
│   ├── model/type.py           # request parameters (model name)
│   ├── predict/
│   │   ├── qwen_inference.py   # loads the base model + LoRA adapter
│   │   └── t5_inference.py     # loads a fine-tuned T5 checkpoint
│   └── services/auth.py        # API-key check (API_KEY from .env)
├── notebooks/
│   ├── qwen_text_summarizer.ipynb
│   └── t5_small_text_summarizer.ipynb
└── requirements.txt
```


## Getting started

### 1. Reproduce the experiments

1. Open a notebook in Google Colab (GPU runtime). The notebooks were run on Colab.
2. Run the cells in order. Each notebook installs its own extra dependencies (for example `trl`, `peft` and `bitsandbytes` for Qwen).
3. The notebook prints the ROUGE scores and saves the fine-tuned model.

### 2. Run the API locally (work in progress)

```bash
pip install -r requirements.txt
pip install uvicorn
```

Create a `.env` file in the project root:
> [!NOTE]
>  I built this API for my own personal project, which is why it is protected by an API key.

```
API_KEY=your-secret-key
```

Place the trained weights in `models/` (see above), then start the service from the project root:

```bash
python -m api.app
```

The service listens on `http://127.0.0.1:8000`, and the interactive docs are available at `/docs`. A single `GET /summarize` endpoint takes the text to summarise and the model to use (`t5-v1`, `t5-v2` or `qwen`), protected by an API key.

## Status and roadmap

The notebooks and results above are complete. The API is under development and **has not been tested end-to-end yet**.

- [ ] Test the `/summarize` endpoint for each model and fix any issues found
- [ ] Add automated tests for the API and the authentication check
- [ ] Record T5 latency and memory use to compare resource cost with Qwen
- [ ] Add qualitative examples (inputs, references, and predictions side by side)
