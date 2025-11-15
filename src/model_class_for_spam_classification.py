import torch
import tiktoken

class SpamClassifier:
    def __init__(self, GPTModel, model_path):
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1280,
            "n_heads": 20, 
            "n_layers": 36,
            "drop_rate": 0.1,
            "qkv_bias": True
        }
        
        self.model = GPTModel(BASE_CONFIG)
        self.model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=False)
        self.model.eval()
        self.tokenizer = tiktoken.get_encoding('gpt2')
    
    def __call__(self, text):
        tokens = self.tokenizer.encode(text)[:256] + [50256] * (256 - len(self.tokenizer.encode(text)[:256]))
        inputs = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            pred = torch.argmax(outputs[:, -1, :]).item()
            
        return 'spam' if pred else 'not_spam'