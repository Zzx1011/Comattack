import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser
from peft import LoraConfig
from safetensors.torch import load_file
from typing import List, Dict, final
from utils import  generate_output_from_attacked_suffix, generate_output_from_decoder_memory

class ICAERecommendationAttacker:
    def __init__(self, 
                 model: ICAE, 
                 demos: Dict[str, str], 
                 target_demo_key: str, 
                 modified_target_text: str,
                 best_demo_key: str,
                 suffix_len: int = 5,
                 num_steps: int = 500,
                 lr: float = 1e-2,
                 device: str = "cuda"):

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = model.tokenizer
        self.device = device
        self.suffix_len = suffix_len
        self.num_steps = num_steps
        self.lr = lr

        self.demos = demos
        self.target_demo_key = target_demo_key
        self.best_demo_key = best_demo_key
        self.modified_target_text = modified_target_text

        self.target_input_ids = self.tokenizer(self.demos[target_demo_key], truncation=True, padding=False, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            self.target_embeddings = model.tokens_to_embeddings(self.target_input_ids)

        # Compute target embedding based on modified version (with inserted keywords)
        modified_ids = self.tokenizer(modified_target_text, truncation=True, padding=False, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            self.target_embedding = model._compress(modified_ids).detach()

        # Compute best demo's embedding (as target reference)
        best_ids = self.tokenizer(demos[best_demo_key], truncation=True, padding=False, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():
            self.best_embedding = model._compress(best_ids).detach()

        # Soft suffix logits (trainable)
        vocab_size, embed_dim = model.get_input_embeddings().weight.shape
        self.token_logits = torch.randn((suffix_len, vocab_size), requires_grad=True, device=device)
        self.optimizer = Adam([self.token_logits], lr=lr)

    def run(self):
        for step in tqdm(range(self.num_steps)):
            self.optimizer.zero_grad()

            # Construct soft suffix
            token_probs = F.softmax(self.token_logits, dim=-1).to(self.model.icae.dtype)
            soft_suffix = token_probs @ self.model.get_input_embeddings().weight  # [suffix_len, D]
            combined_embed = torch.cat([self.target_embeddings, soft_suffix.unsqueeze(0)], dim=1)  # [1, L+S, D]

            # Compress attacked target demo with soft suffix
            memory_attacked = self.model._compress(inputs_embeds=combined_embed)

            # Loss: minimize distance to best demo
            loss = -F.cosine_similarity(memory_attacked, self.best_embedding, dim=-1).mean()
            loss.backward()
            self.optimizer.step()

            if step % 10 == 0:
                print(f"[Step {step}] Cosine similarity: {-loss.item():.4f}")

        return self.finalize()

    def finalize(self):
        with torch.no_grad():
            probs = F.softmax(self.token_logits, dim=-1)
            token_ids = torch.argmax(probs, dim=-1).tolist()
            suffix_text = self.tokenizer.decode(token_ids)
            print("\n[Final Suffix]:", suffix_text)
            return suffix_text, token_ids

    def generate_decoder_input(self, suffix_token_ids: List[int]) -> torch.Tensor:
        """
        Compress all demos (insert suffix into target_demo) and return memory list
        """
        compressed_memories = []
        for key, text in self.demos.items():
            if key == self.target_demo_key:
                modified_ids = self.tokenizer(text, truncation=True, return_tensors='pt')['input_ids'][0].to(self.device)
                attacked_ids = torch.cat([modified_ids, torch.LongTensor(suffix_token_ids).to(self.device)], dim=0).unsqueeze(0)
                memory = self.model._compress(attacked_ids)
            else:
                ids = self.tokenizer(text, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                memory = self.model._compress(ids)
            # print(f"[Demo {key}] Memory shape: {memory.shape}")
            compressed_memories.append(memory)
        return torch.cat(compressed_memories, dim=0)  # [1, num_demo, D]


if __name__ == "__main__":
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

    demos = {
        "demo_1": "Tesla Model Y (7-seat option) \u2013 The Model Y\u2019s optional third row expands seating to seven, while the Long Range version delivers an EPA-estimated 330 miles. Supercharger network access ensures fast, reliable charging on long trips. Safety features include Autopilot with Full Self-Driving upgrade capability, a rigid center structure, and eight airbags. The spacious cargo area (68 cubic feet with seats folded) pairs with rear USB-C ports, roof-mounted air vents, and heated second-row seats for passenger comfort.",
        "demo_2": "Mercedes-Benz EQB \u2013 Starting around $55,000, the EQB offers an optional third-row for two extra seats and up to 260 miles of range. Its 11 kW onboard charger supports Level 2 charging, and DC fast-charging can add about 80 miles in 25 minutes. Standard driver-assist tech includes Active Distance Assist DISTRONIC and Active Steering Assist. Interior appointments feature three-zone climate control, ambient lighting, and abundant cupholders and USB ports across all three rows.",
        "demo_3": "Volvo XC90 Recharge \u2013 The plug-in hybrid option combines electric-only driving of up to 30 miles with a turbo- and supercharged powertrain for seamless transitions. Seating for seven, with 85 cubic feet of cargo space (rear seats folded), and top-tier safety systems\u2014Pilot Assist and Run-off Road Mitigation\u2014ensure family security. Heated, ventilated front seats, power-folding third-row seats, and a Bowers & Wilkins premium sound system elevate passenger comfort for any journey.",
        "demo_4": "Kia EV9 \u2013 With a three-row layout seating up to seven and an EPA range projected at 300 miles, the EV9 blends space and efficiency. Ultra-fast 800V charging capability adds approximately 70 miles in just six minutes. Standard Highway Driving Assist II, remote smart parking assist, and blind-spot collision-avoidance make driving stress-free. Inside, second-row captain\u2019s chairs, a panoramic dual curved display, and rear-seat entertainment options keep all passengers happy.",
        "demo_5": "Rivian R1S \u2013 This adventure-ready SUV seats up to seven and offers around 316 miles of range in its larger battery configuration. DC fast-charging at up to 200 kW can restore 140 miles in 20 minutes. Driver-assist features include adaptive cruise control, lane-keep assist, and off-road driving modes. The R1S boasts a premium interior with quilted leather seats, a 15.6\u2033 touchscreen, integrated storage bins, and individual climate zones for each row."
    }

    attacker = ICAERecommendationAttacker(
        model=model,
        demos=demos,
        target_demo_key="demo_4",
        modified_target_text="Kia EV9 with seven seats, 300-mile range, ample cargo, top safety, and rear luxury features",
        best_demo_key="demo_1",
        suffix_len=5,
        num_steps=50,
        lr=1e-2,
        device="cuda"
    )

    final_suffix, token_ids = attacker.run()
    final_memory_input = attacker.generate_decoder_input(token_ids)

    # You may now pass `final_memory_input` into your LLM decoder pipeline.
    print("\n[Final memory input shape]:", final_memory_input.shape)

    final_memory_input = final_memory_input.unsqueeze(0).to("cuda")
    generated_answer = generate_output_from_decoder_memory(
    model=model,
    decoder_memory_embeddings=final_memory_input,
    prompt_text="My family needs a three-row electric vehicle for daily school runs and road trips. Which product would you recommend for the following needs?",
    max_out_length=128,
    device="cuda"
    )

    print("\n[Final Answer]:", generated_answer)