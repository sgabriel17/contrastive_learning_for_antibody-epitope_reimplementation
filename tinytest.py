import torch
from transformers import AutoTokenizer
from models.AbLangRBD1.ablangpaired_model import AbLangPaired, AbLangPairedConfig

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_dir = "models/AbLangRBD1"

config = AbLangPairedConfig(checkpoint_filename=f'{model_dir}/model.safetensors')
model = AbLangPaired(config, device=device).to(device)
model.eval()

heavy_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/heavy_tokenizer")
light_tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/light_tokenizer")

hc = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISWNSGSIGYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAK"
lc = "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYNTNNLQTGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGAGTKLEIK"

h = heavy_tokenizer(" ".join(hc), return_tensors="pt")
l = light_tokenizer(" ".join(lc), return_tensors="pt")

with torch.no_grad():
    emb = model(
        h_input_ids=h["input_ids"].to(device),
        h_attention_mask=h["attention_mask"].to(device),
        l_input_ids=l["input_ids"].to(device),
        l_attention_mask=l["attention_mask"].to(device),
    )

print(emb.shape)