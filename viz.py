import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")

def predict_next(text):
    input_ids = torch.tensor([tokenizer.encode(text)])
    output = model(input_ids)
    
    logits = output.logits[0, -1]
    
    probs = torch.softmax(logits, dim=0)
    
    top = torch.topk(probs, 5)
    
    results = []
    for i in range(5):
        word = tokenizer.decode(top.indices[i])
        prob = top.values[i].item()
        results.append((word, prob))
    return results

def get_attention(text):
    input_ids = torch.tensor([tokenizer.encode(text)])
    outputs = model(input_ids, output_attentions=True)
    return outputs.attentions

def show_attention(text, query_word, layer, head):
    attentions = get_attention(text)
    grid = attentions[layer][0, head] 
    tokens = tokenizer.encode(text)
    words = [tokenizer.decode(t) for t in tokens]
    
    word = " " + query_word
    
    pos = words.index(word)
    
    row = grid[pos]
    
    for i in range(len(row)):
        print(words[i], row[i].item())
    
    
    

show_attention("The cat sat on the", "on", 5, 10)