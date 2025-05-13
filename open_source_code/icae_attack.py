import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from transformers import HfArgumentParser
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig
from safetensors.torch import load_file


class ICAEAttackerBase:
    def __init__(self, model, base_text, suffix_len=5, num_steps=1000, lr=1e-2, device="cuda"):
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        lora_config = LoraConfig(
            r=512,
            lora_alpha=32,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = ICAE(model_args, training_args, lora_config)
        state_dict = load_file(training_args.output_dir)
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.device = device
        self.suffix_len = suffix_len
        self.num_steps = num_steps
        self.lr = lr

        # Prepare base input and embeddings
        self.base_input_ids = self.tokenizer(base_text, truncation=True, max_length=5120, padding=False)['input_ids']
        self.base_input_ids = torch.LongTensor([self.base_input_ids]).to(device)
        with torch.no_grad():
            self.base_embeds = self.model.tokens_to_embeddings(self.base_input_ids)

        # Initialize trainable token logits
        vocab_size, _ = self.model.get_input_embeddings().weight.shape
        self.token_logits = torch.randn((suffix_len, vocab_size), requires_grad=True, device=device)
        self.optimizer = Adam([self.token_logits], lr=lr)

    def step(self):
        raise NotImplementedError("Must be implemented by subclasses.")

    def run(self):
        for step in tqdm(range(self.num_steps)):
            self.optimizer.zero_grad()
            loss = self.step()
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print(f"[Step {step}] Loss: {loss.item():.4f}")
        return self.finalize()

    def finalize(self):
        with torch.no_grad():
            token_probs = F.softmax(self.token_logits, dim=-1)
            token_ids = torch.argmax(token_probs, dim=-1).tolist()
            decoded_suffix = self.tokenizer.decode(token_ids)
            print(f"\n[Final Decoded Suffix]: {decoded_suffix}")
            return decoded_suffix, token_ids
        
class TargetedICAEAttacker(ICAEAttackerBase):
    def __init__(self, model, base_text, target_text, suffix_len=5, num_steps=1000, lr=1e-2, device="cuda"):
        super().__init__(model, base_text, suffix_len, num_steps, lr, device)

        # Compute target embedding
        target_input_ids = self.tokenizer(target_text, truncation=True, max_length=5120, padding=False)['input_ids']
        target_input_ids = torch.LongTensor([target_input_ids]).to(device)
        with torch.no_grad():
            self.target_embedding = self.model._compress(target_input_ids).detach()

    def step(self):
        token_probs = F.softmax(self.token_logits, dim=-1).to(self.model.icae.dtype)
        soft_suffix = token_probs @ self.model.get_input_embeddings().weight
        combined = torch.cat([self.base_embeds, soft_suffix.unsqueeze(0)], dim=1)
        memory_slot = self.model._compress(inputs_embeds=combined)
        loss = -F.cosine_similarity(memory_slot, self.target_embedding.unsqueeze(0), dim=-1).mean()
        return loss
    
class NonTargetedICAEAttacker(ICAEAttackerBase):
    def __init__(self, model, base_text, suffix_len=5, num_steps=1000, lr=1e-2, device="cuda"):
        super().__init__(model, base_text, suffix_len, num_steps, lr, device)
        with torch.no_grad():
            self.base_embedding = self.model._compress(self.base_input_ids).detach()

    def step(self):
        token_probs = F.softmax(self.token_logits, dim=-1).to(self.model.icae.dtype)
        soft_suffix = token_probs @ self.model.get_input_embeddings().weight
        combined = torch.cat([self.base_embeds, soft_suffix.unsqueeze(0)], dim=1)
        memory_slot = self.model._compress(inputs_embeds=combined)
        loss = F.cosine_similarity(memory_slot, self.base_embedding, dim=-1).mean()
        return loss  # maximize distance from base == minimize similarity
    
device = "cuda"

# Load model and arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

lora_config = LoraConfig(
    r=512,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = ICAE(model_args, training_args, lora_config)
state_dict = load_file(training_args.output_dir)
model.load_state_dict(state_dict, strict=False)

base_text = "I don't have a favorite condiment..."
target_text = "This is the target text that we want the suffix to mimic."

# 🧪 Option 1: Targeted attack
attacker = TargetedICAEAttacker(model, base_text, target_text, suffix_len=5, num_steps=500, device=device)

# 🧪 Option 2: Non-targeted attack
# attacker = NonTargetedICAEAttacker(model, base_text, suffix_len=5, num_steps=500, device=device)

decoded_suffix, token_ids = attacker.run()