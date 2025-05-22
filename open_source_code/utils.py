import json
import torch
from tqdm import tqdm


@torch.no_grad()
def generate_output_from_attacked_suffix(
    model,
    base_text: str,
    suffix_token_ids: list[int],
    prompt_text: str,
    max_out_length: int = 128,
    device: str = "cuda"
):
    # model = model.to(device)
    model.eval()
    tokenizer = model.tokenizer

    # 1. base input
    tokenized_input = tokenizer(base_text, truncation=True, max_length=5120, padding=False, return_attention_mask=False)
    input_ids = torch.LongTensor([tokenized_input['input_ids']]).to(device)

    # 2. memory_slots
    base_embeds = model.tokens_to_embeddings(input_ids)
    suffix_ids = torch.LongTensor([suffix_token_ids]).to(device)
    suffix_embeds = model.tokens_to_embeddings(suffix_ids)
    attacked_embeds = torch.cat([base_embeds, suffix_embeds], dim=1)

    memory_slots = model._compress(inputs_embeds=attacked_embeds)

    # 3. decoder input embeddings
    prompt_left_ids = torch.LongTensor([[1, 733, 16289, 28793]]).to(device)
    prompt_right_ids = [model.ft_token_id] + tokenizer(prompt_text, add_special_tokens=False)['input_ids'] + [733, 28748, 16289, 28793]
    prompt_right_ids = torch.LongTensor([prompt_right_ids]).to(device)

    prompt_left_embs = model.tokens_to_embeddings(prompt_left_ids)
    prompt_right_embs = model.tokens_to_embeddings(prompt_right_ids)
    memory_slots = memory_slots.to(prompt_right_embs)

    decoder_input_embeddings = torch.cat([prompt_left_embs, memory_slots.unsqueeze(0), prompt_right_embs], dim=1)

    # 4. decoder input
    output = decoder_input_embeddings.clone()
    generate_text = []
    past_key_values = None

    for _ in range(max_out_length):
        with model.icae.disable_adapter():  
            out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)

        logits = out.logits[:, -1, :model.vocab_size-1]
        past_key_values = out.past_key_values

        next_token_id = torch.argmax(logits, dim=-1)
        # print("Step:", _)
        # print("next_token_id: ", next_token_id)
        if next_token_id.item() == 2:  # EOS token
            break

        token_embed = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
        output = token_embed
        generate_text.append(next_token_id.item())

    return tokenizer.decode(generate_text)

@torch.no_grad()
def generate_output_from_decoder_memory(
    model,
    decoder_memory_embeddings,  # shape: (1, demo_len, hidden_dim)
    prompt_text,
    max_out_length=128,
    device="cuda"
):
    tokenizer = model.tokenizer
    model.eval()

    # # Encode prompt text to embeddings
    # prompt_input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    # with torch.no_grad():
    #     prompt_embeddings = model.icae.get_base_model().model.embed_tokens(prompt_input_ids)
    
    prompt_left_ids = torch.LongTensor([[1, 733, 16289, 28793]]).to(device)
    prompt_right_ids = [model.ft_token_id] + tokenizer(prompt_text, add_special_tokens=False)['input_ids'] + [733, 28748, 16289, 28793]
    prompt_right_ids = torch.LongTensor([prompt_right_ids]).to(device)

    prompt_left_embs = model.tokens_to_embeddings(prompt_left_ids)
    prompt_right_embs = model.tokens_to_embeddings(prompt_right_ids)

    # Concatenate demo memory and prompt embeddings
    decoder_input = torch.cat([prompt_left_embs, decoder_memory_embeddings, prompt_right_embs], dim=1).to(model.icae.dtype)  

    output = decoder_input.clone()
    generated_token_ids = []
    past_key_values = None

    for i in range(max_out_length):
        with model.icae.disable_adapter():
            out = model.icae(
                inputs_embeds=output,
                past_key_values=past_key_values,
                use_cache=True
            )

        logits = out.logits[:, -1, :model.vocab_size - 1]
        past_key_values = out.past_key_values

        next_token_id = torch.argmax(logits, dim=-1)
        # print("Step:", i)
        # print("next_token_id: ", next_token_id)
        # print("eos_token_id: ", tokenizer.eos_token_id)
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_token_ids.append(next_token_id.item())

        next_token_embed = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
        output = next_token_embed
        # print("output shape:", output.shape)
        # print("output: ", output)

    return tokenizer.decode(generated_token_ids, skip_special_tokens=True)