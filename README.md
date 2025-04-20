<p align="center">
  <img src="./assets/logo.png" alt="Project Logo" width="200"/> 
  <!-- Adjust width as needed -->
</p>

# CS776 Project: Reasoning on a Toaster: Enhancing Small Vision-Language Models via Simplified Chain-of-Thought Fine-tuningReasoning Vision 

#### Team
- Aniket 
- Anuj
- Apoorva
- Divyansh
- Rajeev
- Sandeep


## File Structure
```
.
├── assets
│   ├── elo_plot.png
│   ├── sft_256M_plot.png
│   └── sft_500M_plot.png
├── checkpoints
│   ├── SFT_256M_smolcot_mini_3k
│   ├── SFT_256M_smolcot_mini_3k
|   ├──  ....
│   └── SFT_500M_smolcot_massive_3k
├── data
│   ├── Data_cot_traces.ipynb
│   └── filter_data.ipynb
├── eval
│   ├── eval_outputs.ipynb
│   ├── generate_responses.py
│   └── human_eval.ipynb
├── Final_ppt.pdf
├── Midterm_ppt.pdf
├── Project_Report.pdf
├── README.md
├── requirements.txt
├── run.ipynb
└── train
    ├── DPO_Dataset_prep.ipynb
    ├── DPO_VLM_new_smolcot_with_eval.ipynb
    ├── GRPO_gsm8k.py
    └── SFT_vlm_smolcot.py
```

## Environment Setup
First of all setup the python environment required by this project
```
python3 -m venv .venv
source .venv/bin/activate
```
Then install all the dependencies
```
pip install -r requirements.txt
```

## Train
### DPO
- [DPO_Dataset_prep.ipynb](train/DPO_Dataset_prep.ipynb) : Generates response required by DPO. Just run it cell by cell, extra information is present in the markdowns in between.
- [DPO_VLM_new_smolcot_with_eval.ipynb](train/DPO_VLM_new_smolcot_with_eval.ipynb): Runs the DPO. Just run it cell by cell, extra information is present in the markdowns in between. 

### GRPO
- [GRPO_gsm8k.py](train/GRPO_gsm8k.py): Simply applies grpo on smolVLM model with `gsm8k` dataset. Instructions to run 
```bash
python3 GRPO_gsm8k.py
```

### SFT
- [SFT_vlm_smolcot.py](train/SFT_vlm_smolcot.py): Applies SFT on any model and training data as provided. Instructions to run: 
```bash
python3 SFT_vlm_smolcot.py --model_path "HuggingFaceTB/SmolVLM-256M-Instruct" --data_path "../data/new_train_massive_7k_hf" --model_checkpoint_path "SFT_256M_smolcot_massive_7k" --num_epochs 1 --learning_rate 3e-5
```

Although, the arguments are self-explainatory, here is the description:
```bash
usage: script_name.py [-h] --model_path MODEL_PATH --data_path DATA_PATH
                  --model_checkpoint_path MODEL_CHECKPOINT_PATH --num_epochs
                  NUM_EPOCHS --learning_rate LEARNING_RATE

Fine-tune SmolVLM model

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the pre-trained model
  --data_path DATA_PATH
                        Path to the training data
  --model_checkpoint_path MODEL_CHECKPOINT_PATH
                        Path to save the fine-tuned model
  --num_epochs NUM_EPOCHS
                        Number of training epochs
  --learning_rate LEARNING_RATE
                        Learning rate for training
```

## Eval
- [eval_outputs.ipynb](eval/eval_outputs.ipynb): Automatic evaluation. Runs auy cell, more instructions in the markdonw in between the cells
- [human_eval.ipynb](eval/human_eval.ipynb): Human evaluation via Elo score rating. Run cell by cell, more instructions in the markdonw in between the cells
- [generate_responses.py](eval/generate_responses.py): Script to generate response of any model for any dataset. Instructions to use: 
```python
python3 generate_response.py --checkpoint_path /path/to/checkpoint --output_path /path/to/output.json --device cuda --think_prompt True
```

The arguments are self-explanatory, the `think_prompt` argument is tweark to add "_let's think step by step_" in the prompt or not.

## Data
This has the files to sample datapoints from ChartQA data, then generate reasoning traces via api calls and then finally create a Huggingface version of the dataset. Both [Data_cot_traces.ipynb](data/Data_cot_traces.ipynb) and [filter_data.ipynb](data/filter_data.ipynb) should be just run cell by cell, any specific instructions are in the markdowns in between.


## Miscellaneous
- `run.ipynb` can be used to checkout any model trained or untrained model by executing cell by cell, any specific instruction is in the markdowns in between
- `Checkpoints` should be population by downloading the checkpoints from [here](https://drive.google.com/drive/folders/1rb3MppN3LspXLhhRteCIlapfbabGYodN?usp=sharing)
- `traning data` can also be downloaded from [here](https://drive.google.com/drive/folders/1rb3MppN3LspXLhhRteCIlapfbabGYodN?usp=sharing)


## Datasets
- **Training Sets**: We created two main training datasets
using the simplified CoT format: `smolcot-3k` (approx.
3,000 samples) and `smolcot-7k` (approx. 7,000-8,000
samples). These contained the image, question, the sim-
plified Gemini-generated CoT, and the final answer

- **Test Set**: We used a fixed held-out set of 300 samples
from ChartQA (that were not used for generating training
CoT) for evaluating all model variants. This test set
included the image, question, and the ground-truth final
answer.

## Model Description

- `"+step" prompt` is basically adding "_let's think step by step_" in the question prompt fed into the model

| model code       | model description                                                              | model_path                             |
|------------------|--------------------------------------------------------------------------------|----------------------------------------|
| `256M-base`      | Original SmolVLM-256M-Instruct model.                                          | HuggingFaceTB/SmolVLM-256M-Instruct    |
| `500M-base`      | Original SmolVLM-500M-Instruct model.                                          | HuggingFaceTB/SmolVLM-256M-Instruct    |
| `256M-3k`        | `256M-base` fine-tuned on `smolcot-3k`.                                        | checkpoints/SFT_256M_smolcot_mini_3k   |
| `256M-7k`        | `256M-base` fine-tuned on `smolcot-7k`.                                        | checkpoints/SFT_256M_smolcot_massive_7k|
| `500M-3k`        | `500M-base` fine-tuned on `smolcot-3k`.                                        | checkpoints/SFT_500M_smolcot_mini_3k   |
| `500M-7k`        | `500M-base` fine-tuned on `smolcot-7k`.                                        | checkpoints/SFT_256M_smolcot_massive_7k|