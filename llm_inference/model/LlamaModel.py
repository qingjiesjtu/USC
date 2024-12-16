# LlamaModel.py
from .BasicModel import BasicModel
import transformers
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)
class LlamaModel(BasicModel):
    
    def __init__(self, device, precision, model_path, lora_path=None):
        super().__init__(device, precision, model_path, lora_path)
        
    def init_terminators(self):
        # Llama needs to initialize this specifically
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.gen_kwargs["eos_token_id"] = self.terminators
    def perturbation_based_per_seq(self, messages, target):
        # messages: List of messages, e.g.,
        # [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "question2"}]
        # target: The target text for attribution, e.g., "Yes" or "No"

        # Create placeholder messages with "{}" as content
        placeholder_messages = []
        for message in messages:
            placeholder_messages.append({
                "role": message["role"],
                "content": "{}"
            })

        # Use apply_chat_template to generate the template with placeholders
        # This will give us the template string with "{}" in place of the actual contents
        template = self.tokenizer.apply_chat_template(
            placeholder_messages,
            pad_token_id=self.tokenizer.eos_token_id,
            add_generation_prompt=True,
        )

        # Check what the next token is
        tmp_template = self.tokenizer.apply_chat_template(
            messages,
            pad_token_id=self.tokenizer.eos_token_id,
            add_generation_prompt=True,
        )
        tmp_template_tensor = torch.tensor([tmp_template], dtype=torch.long).to(self.model.device)
        if "llama2" in self.model_path.lower() or "llama-2" in self.model_path.lower():
            neet_to_add = torch.tensor([29871])
            tmp_template_tensor = torch.cat((tmp_template_tensor, neet_to_add))
        # Generate the next token(s) from the model
        tmp = self.pipeline.model.generate(tmp_template_tensor, do_sample=False, max_length=tmp_template_tensor.shape[1] + 1)
        if "llama2" in self.model_path.lower() or "llama-2" in self.model_path.lower():
            print(f"next token: {self.tokenizer.decode(tmp[0][-1])}")
        else:
            print(f"next token: {self.tokenizer.decode(tmp[-1])}")
        
        # Append the special token for Llama 2 if necessary
        if "llama2" in self.model_path.lower() or "llama-2" in self.model_path.lower():
            neet_to_add = torch.tensor([29871])
            template = torch.cat((torch.tensor(template), neet_to_add))
        # Extract the actual contents to fill into the placeholders
        values = [message["content"] for message in messages]
        
        template = self.tokenizer.decode(template)
        
        # Create the TextTemplateInput
        inp = TextTemplateInput(
            template=template,
            values=values,
        )

        # Perform attribution
        fa = FeatureAblation(self.model)
        llm_attr = LLMAttribution(fa, self.tokenizer)
        attr_res = llm_attr.attribute(inp, target=target)
        res = attr_res.seq_attr.cpu().tolist()
        print(res)

        return res
