import json,torch,numpy as np,time,glob
from transformers import AutoTokenizer,AutoModel
d=json.load(open('pairs.json'));texts=d['texts']
P=[p for p in glob.glob('/home/x_yushe/.cache/huggingface/hub/models--microsoft--unixcoder-base/snapshots/*') if glob.glob(p+'/config.json')][0]
tok=AutoTokenizer.from_pretrained(P);model=AutoModel.from_pretrained(P).cuda().eval()
MAX=512
def encode(t):
    ids=tok.convert_tokens_to_ids(tok.tokenize(t if t.strip() else ' ')[:MAX-4])
    ids=[tok.cls_token_id,tok.convert_tokens_to_ids('<encoder-only>'),tok.sep_token_id]+ids+[tok.sep_token_id]
    return ids
order=sorted(range(len(texts)),key=lambda i:len(texts[i]))
out=np.zeros((len(texts),768),dtype=np.float32);t0=time.time()
with torch.no_grad():
    for b in range(0,len(order),64):
        idx=order[b:b+64];ids=[encode(texts[i]) for i in idx];L=max(len(x) for x in ids)
        inp=torch.tensor([x+[tok.pad_token_id]*(L-len(x)) for x in ids]).cuda()
        mask=(inp!=tok.pad_token_id).float().unsqueeze(-1)
        h=model(inp,attention_mask=(inp!=tok.pad_token_id)).last_hidden_state
        out[idx]=((h*mask).sum(1)/mask.sum(1)).cpu().numpy()
        if (b//64)%50==0: print(b,len(order),round(time.time()-t0),flush=True)
np.save('emb_ux.npy',out);print('done',flush=True)
