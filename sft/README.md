# Supervised Finetuning of GPT2 (355M) model

This repo is taken from https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07. The changes here are mainly for self-learning and new experiments.

### Dependencies

Please install the following dependencies to run the code.
```bash
pip install -r requirements.txt
```

### Code execution

- main.py : To finetune the GPT2 model (using pre-trained weights) on the instruction-data.json provided
- infer_text.py : To perform inference and gather responses on the test data.
- dataloader.py : Contains all utilities to process dataset

Usage:

```bash
python main.py
```
This will save the model at the end with the name : `gpt2-medium355M-sft.pth`. The size of the checkpoint is approximately 1.72 GB.

```bash
python infer_text.py
```
This will load the previously saved model and generates responses based on test data into a json file (instruction-data-with-response.json). 

### Evaluation
Please refer to eval_openai.py which uses GPT models from OpenAI to evaluate the generated model responses (obtained from running infer_text.py)

1) Using gpt-3.5-turbo to score GPT2
```bash
Number of entries in the test set: 110
Number of scores: 110 of 110
Average score: 53.25
```

1) Using gpt-4 to score GPT2
```bash
Number of entries in the test set: 110
Number of scores: 110 of 110
Average score: 41.14
```

You can also use offline LLMs like Llama3 to evaluate these responses.

### Notes:

* We pad the inputs with one <eos> token (labelled 50526 - pad token) and `-100` token as well. `-100` is the ignore index used in Pytorch's cross-entropy calculation which ignores the loss calculation for that particular token.

### References:

1) https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07