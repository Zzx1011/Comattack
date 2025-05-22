import json
from sympy import N
from torch.utils.data import Dataset

class BaseMultiOutputDataset(Dataset):
    def __init__(self, json_path, output_keys, tokenizer=None, max_length=512):
        """
        :param json_path: Path to JSON file
        :param output_keys: List of keys to extract outputs (e.g., ['output1', 'output2'])
        :param tokenizer: HuggingFace tokenizer (optional)
        :param max_length: Max token length if using tokenizer
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.output_keys = output_keys
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        question = entry["question"]
        requirement = entry["requirements"]
        outputs = [entry[key] for key in self.output_keys]

        if self.tokenizer:
            encoded = self.tokenizer(
                [question] * len(outputs),  # paired with each output
                outputs,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'],  # shape: (num_choices, seq_len)
                'attention_mask': encoded['attention_mask'],
                'question': question,
                'requirement': requirement,
                'outputs': outputs
            }
        else:
            return {
                'question': question,
                'requirement': requirement,
                'outputs': outputs
            }

class FullMultiOutputDataset(BaseMultiOutputDataset):
    def __init__(self, json_path, tokenizer=None, max_length=512):
        output_keys = ['output1', 'output2', 'output3', 'output4', 'output5']
        super().__init__(json_path, output_keys, tokenizer, max_length)

class PartialMultiOutputDataset(BaseMultiOutputDataset):
    def __init__(self, json_path, tokenizer=None, max_length=512):
        output_keys = ['output1', 'output2']
        super().__init__(json_path, output_keys, tokenizer, max_length)


class FullMultiOutputDatasetWithTarget(Dataset):
    def __init__(self, json_path, tokenizer=None):
        self.tokenizer = tokenizer
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.samples = []
        for entry in self.data:
            question = entry.get("question", "")
            
            # 获取所有 requirement 字段
            requirements = [
                value for key, value in entry.items()
                if key.startswith("requirement_")
            ]
            
            # 获取所有 demo_X 候选项
            demos = [entry[f"demo_{i}"] for i in range(1, 6) if f"demo_{i}" in entry]
            
            best_key = entry.get("best")
            target_key = entry.get("target")
            # print(f"best_key: {best_key}, target_key: {target_key}")

            best_idx = int(best_key.split("_")[1]) - 1 if best_key else None
            target_idx = int(target_key.split("_")[1]) - 1 if target_key else None

            self.samples.append({
                "question": question,
                "requirements": requirements,
                "demos": demos,
                "best": demos[best_idx] if best_idx is not None else None,
                "target": demos[target_idx] if target_idx is not None else None
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Example usage
    dataset = FullMultiOutputDatasetWithTarget("/root/datasets/data_with_target.json", tokenizer=None)
    sample = dataset[0]
    print(sample.keys())  
    print(sample['question'])
    print(sample['requirements'])
    print(sample['demos'])
    print(sample['best'])
    print(sample['target'])
