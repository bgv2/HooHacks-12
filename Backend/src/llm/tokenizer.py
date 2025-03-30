from transformers import AutoTokenizer

def load_llama3_tokenizer():
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer

def tokenize_text(text: str, tokenizer) -> list:
    tokens = tokenizer.encode(text, return_tensors='pt')
    return tokens

def decode_tokens(tokens: list, tokenizer) -> str:
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text