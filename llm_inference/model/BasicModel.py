# BasicModel.py
import transformers
import torch
from peft import PeftModel, LoraConfig, TaskType

class BasicModel:

    def __init__(self, device, precision, model_path, lora_path=None):
        self.device = device
        self.precision = precision
        self.model_path = model_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.lora_path = lora_path
        
        if self.lora_path is None:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=getattr(torch, precision), 
                trust_remote_code=True,
                device_map="auto",
            )
        elif self.lora_path is not None:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, precision),
                device_map="auto",
            )
            config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                inference_mode=True,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1
            )
            self.model.load_adapter(self.lora_path, peft_config = config)

        self.model.eval()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.gen_kwargs = {}
        self.init_terminators()
        
    def init_terminators(self):
        pass
    
    def generate(self, messages, max_new_tokens=None, not_do_sample = False ,top_k=None, top_p=None, temperature=None):
        if max_new_tokens is not None:
            self.gen_kwargs["max_new_tokens"] = max_new_tokens
        if not_do_sample is None:
            pass
        elif not_do_sample == True:
            self.gen_kwargs["do_sample"] = False
        elif not_do_sample == False:
            self.gen_kwargs["do_sample"] = True

        if top_k is not None:
            self.gen_kwargs["top_k"] = top_k
        if top_p is not None:
            self.gen_kwargs["top_p"] = top_p
        if temperature is not None:
            self.gen_kwargs["temperature"] = temperature
            
        if "pretrain" in self.model_path:
            outputs = self.pipeline(messages[0]['content'], **self.gen_kwargs)
            messages_output = outputs[0]['generated_text']
            return messages_output
        else:
            outputs = self.pipeline(messages, **self.gen_kwargs)
            messages_output = outputs[0]['generated_text']
            
            return messages_output[-1]['content']
        
    def generate_with_probs(self, messages, max_new_tokens=None, not_do_sample=False, top_k=None, top_p=None, temperature=None):
        """
        Generate text along with the probabilities for each generated token.

        Args:
            messages (list): A list of messages to generate a response for.
        Returns:
            generated_tokens (list of str): List of generated tokens as strings.
            probabilities (list of float): List of probabilities corresponding to each generated token.
        """
        if max_new_tokens is not None:
            self.gen_kwargs["max_new_tokens"] = max_new_tokens
        if not_do_sample is not None:
            self.gen_kwargs["do_sample"] = not not_do_sample
        if top_k is not None:
            self.gen_kwargs["top_k"] = top_k
        if top_p is not None:
            self.gen_kwargs["top_p"] = top_p
        if temperature is not None:
            self.gen_kwargs["temperature"] = temperature

        # Add parameters to get output token probabilities
        self.gen_kwargs["output_scores"] = True
        self.gen_kwargs["return_dict_in_generate"] = True

        if "pretrain" in self.model_path:
            raise NotImplementedError("Pretrained models are not supported for this method.")
        else:
            # Prepare the inputs
            inputs = self.pipeline.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.model.device)
            
            # Generate outputs
            outputs = self.pipeline.model.generate(
                **inputs,
                **self.gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Get generated token IDs (excluding the input tokens)
            generated_token_ids = outputs.sequences[:, inputs['input_ids'].shape[-1]:][0].tolist()
            
            # Decode token IDs to strings
            generated_tokens = [
                self.pipeline.tokenizer.decode([token_id], skip_special_tokens=True) 
                for token_id in generated_token_ids
            ]

            # Get probabilities for each generated token
            probabilities = []
            for i, score in enumerate(outputs.scores):
                prob = torch.softmax(score, dim=-1)
                token_id = generated_token_ids[i]
                token_prob = prob[0, token_id].item()
                probabilities.append(token_prob)
            
            return generated_tokens, probabilities

    def generate_one_token_probs(self, messages):
        """
        Generate one token and return a dict contain the probabilities of the token
        with the highest probability, 'Yes', 'No', the token with the higher
        probability between 'Yes' and 'No'.

        Args:
            messages (list): A list of messages to generate a response for.

        Returns:
            result_dict (dict): A dictionary containing the following
                - max_prob_token (str): The token with the highest probability.
                - max_prob (float): The probability of the token with the highest
                probability.
                - Yes_prob (float): The probability of the token 'Yes'.
                - No_prob (float): The probability of the token 'No'.
                - higher_prob_token (str): The token with the higher probability
                between 'Yes' and 'No'.
                - higher_prob (float): The probability of the token with the higher
                probability.
        """
        # Copy gen_kwargs to avoid modifying the original dictionary
        gen_kwargs = self.gen_kwargs.copy()
        # Set generation parameters
        gen_kwargs.update({
            "max_new_tokens": 1,
            "do_sample": False,  # Greedy decoding
            "output_scores": True,
            "return_dict_in_generate": True
        })

        if "pretrain" in self.model_path:
            raise NotImplementedError("Pretrained models are not supported for this method.")
        else:
            # Prepare the inputs
            
            inputs = self.pipeline.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True # mark
            ).to(self.model.device)
            
            if "llama2" in self.model_path or "Llama-2" in self.model_path:
                # 加上token 
                token_to_add = torch.tensor([[29871]]).to(self.model.device)
                inputs = torch.cat([inputs, token_to_add], dim=1)
            # Generate outputs
            outputs = self.pipeline.model.generate(
                inputs,
                **gen_kwargs,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Get logits at the generated step
            logits = outputs.scores[0]  # Shape: (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # Shape: (batch_size, vocab_size)

            # Get the token ID with the highest probability
            max_prob_token_id = torch.argmax(probs, dim=-1)  # Shape: (batch_size,)
            max_prob_token = self.pipeline.tokenizer.decode(max_prob_token_id, skip_special_tokens=True)

            # Get the token IDs for 'Yes' and 'No'
            yes_token = 'Yes'
            no_token = 'No'

            yes_token_id = self.pipeline.tokenizer.encode(yes_token, add_special_tokens=False)
            no_token_id = self.pipeline.tokenizer.encode(no_token, add_special_tokens=False)

            if len(yes_token_id) != 1 or len(no_token_id) != 1:
                raise ValueError("Tokenizer does not encode 'Yes' or 'No' as single tokens.")

            yes_token_id = yes_token_id[0]
            no_token_id = no_token_id[0]

            # Get probabilities for 'Yes' and 'No'
            yes_prob = probs[0, yes_token_id].item()
            no_prob = probs[0, no_token_id].item()

            # Determine which one has higher probability
            if yes_prob > no_prob:
                higher_prob_token = 'Yes'
            else:
                higher_prob_token = 'No'

            result_dict = {
                'max_prob_token': max_prob_token,
                'max_prob': probs[0, max_prob_token_id].item(),
                'Yes_prob': yes_prob,
                'No_prob': no_prob,
                'higher_prob_token': higher_prob_token,
                'higher_prob': yes_prob if higher_prob_token == 'Yes' else no_prob
            }

            return result_dict
