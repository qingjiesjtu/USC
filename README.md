# Understanding the Dark Side of LLMs’ Intrinsic Self-Correction

This is the code repository of our anonymous submission: Understanding the Dark Side of LLMs’ Intrinsic Self-Correction.
This repository contains the codes for:
1. Evaluating LLMs' intrinsic self-correction on Yes/No question answering task.
2. Evaluating LLMs' final answer wavering in a 10-round conversation.
3. Probing LLMs' internal answers.
4. Interpreting LLMs' prompt bias: Prompt Attribution and Contribution Tracking (PACT).

Please also check our anonymous [project website](https://x-isc.github.io).

## Setup

### Environment Setup
1. Create a new conda environment:
```bash
conda create -n self-correction python=3.9 -y  # Python >= 3.9 required
conda activate self-correction
```

2. Install PyTorch:
- Visit [PyTorch's website](https://pytorch.org/get-started/locally/) to install PyTorch 2.5.1 with the configuration matching your hardware.

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### API Configuration
- For OpenAI models: Configure `OPENAI_API_KEY` and `OPENAI_ENDPOINT` in `llm_inference/api_config.py`
- For local models: Set up your [Hugging Face access token](https://huggingface.co/docs/hub/index) or login to huggingface.co, and than you can use the default model path


## 1. Evaluating LLMs' intrinsic self-correction on Yes/No question answering task.
Reproduce Table 1: Self-correction performance on Yes/No question answering tasks

```bash
python run_self_correction.py --model llama3-8b-instruct --devices 0
```

Available models:
- llama2-7b-instruct
- llama3-8b-instruct
- llama3.1-8b-instruct
- gpt3.5-turbo
- gpt4o
- o1-preview
- o1-mini

--devices: GPU ids, eg: "0,1" to use two or more GPUs or "0" to use the first GPU

Results will be saved in `./results/$MODEL_NAME/self_correction/`

To generate the summary table for all models:
```bash
python draw/metric.py  # Outputs to ./metric.csv
```

## 2. Evaluating LLMs' final answer wavering in a 10-round conversation.
Reproduce Figure 2: Analysis of how LLMs change their final answers in a 10-round conversation

```bash
python run_self_correction.py --model $MODEL_NAME --devices $DEVICES --repeat_exp --rounds 10
```

To generate the visualization:
```bash
python draw/change_answer_times.py  # Outputs to ./model_answer_change.pdf
```

## 3. Probing LLMs' internal answers.
Reproduce Figure 3: Analysis of internal answer changes during self-correction

```bash
# With attack
python run_lens.py --model llama3-8b-instruct --devices $DEVICES --exp tuned_lens

# Round 0: without attack
python run_lens.py --model llama3-8b-instruct --devices $DEVICES --exp tuned_lens --round 0
```

Base on the results, you can generate the visualizations:
```bash
# Figure 3 Left: Internal answer wavering
python draw/case_lens_internal_answer_wavering.py --model llama3-8b-instruct

# Figure 3 Right: "Are you sure?" vs. "You are wrong"
python draw/average_layer_confidence.py --model llama3-8b-instruct
```

Note: Use `--devices` to specify GPU devices (e.g., "0,1" for multiple GPUs, "0" for single GPU)

## 4. Interpreting LLMs' prompt bias: Prompt Attribution and Contribution Tracking (PACT).

Reproduce Figure 4: Analysis of prompt bias during self_correction

Follow our tutorial `./pact.ipynb` to generate PACT visualization.