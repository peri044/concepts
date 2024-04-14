# Direct Preference Optimization
- This is an LLM alignment algorithm which is simpler alternative to RLHF. 
- There's no explicit reward model in this algorithm unlike RLHF-PPO

## Terminology
pi_ref and pi are being used in the paper and equations. They refer to the policies which is the model itself. pi_ref is the frozen model obtained after SFT. pi is the model being 
currently trained to align with human preferences.

## Inputs to DPO algorithm
The inputs to DPO are a) input_query b) chosen_response c) rejected_response
You get the following from the input batch (Code reference: https://github.com/NVIDIA/NeMo-Aligner/blob/9db62d6d8daf5046825f9dbb38a13b35af881515/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L112-L121)
- input tokens for chosen and rejected
- labels for chosen and rejected (which is 1 token ahead of input tokens)
- ref_logprobs for chosen and rejected

You do a forward pass on your GPT model (with input_tokens) and calculate the loss

## DPO loss function
For loss calculation, you need 
- chosen response token log probs 
- rejected response token log probs
- reference_model chosen/rejected response token probs (the base SFT model)

L_dpo = -logsigmoid(chosen_rewards - rejected_rewards)

Note: Rewards for chosen responses should be higher than the rejected ones. So the logsigmoid should be higher which means minimizing L_dpo should achieve that objective.

rewards = chosen_rewards - rejected_rewards

chosen_rewards = pi_logprobs - pi_ref_log_probs (for chosen responses)
rejected_rewards = pi_logprobs - pi_ref_log_probs (for rejected responses)

Code reference: <a href="https://github.com/NVIDIA/NeMo-Aligner/blob/9db62d6d8daf5046825f9dbb38a13b35af881515/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L194-L199">https://github.com/NVIDIA/NeMo-Aligner/blob/9db62d6d8daf5046825f9dbb38a13b35af881515/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L194-L199</a>

## Resources
1) https://magazine.sebastianraschka.com/p/tips-for-llm-pretraining-and-evaluating-rms
2) https://github.com/NVIDIA/NeMo-Aligner/blob/9db62d6d8daf5046825f9dbb38a13b35af881515/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py
3) https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/dpo.html



