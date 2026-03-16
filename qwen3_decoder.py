import time

import torch
import torch.nn as nn


from gptq import *
from modelutils import *
from evalutor import *
#!pip install -U qwen-asr

from evalutor import *
import evaluate as evaluate_lib
from hy_datautils import build_cali_dataloader
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_qwen3(model_name):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=None,
        max_inference_batch_size=32,
        max_new_tokens=256,
    )

    return model

@torch.no_grad()
def qwen3ASR_sequential(model, dataloader, dev):
    print('Starting ...')
    
    decoder = model.model.thinker.model
    layers = decoder.layers
    config = decoder.config
    
    use_cache = getattr(config, "use_cache", False)
    if hasattr(config, "use_cache"):
        config.use_cache = False
        
    decoder.embed_tokens = decoder.embed_tokens.to(dev)
    decoder.norm = decoder.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = decoder.config.hidden_size
    inps = torch.zeros(
        (args.nsamples, model.seqlen, hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    inps = []
    attn_masks = []
    pos_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach())
            attn_masks.append(kwargs.get("attention_mask", None))
            pos_ids.append(kwargs.get("position_ids", None))
            cache["i"] += 1
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch)
        except ValueError:
            pass
        if cache["i"] >= args.nsamples:  # 추가
            break

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    decoder.embed_tokens = decoder.embed_tokens.cpu()
    decoder.norm = decoder.norm.cpu()
    torch.cuda.empty_cache()
    
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
                
            for j in range(args.nsamples):
                kwargs = {}
                if attention_mask is not None:
                    kwargs["attention_mask"] = attention_mask
                if position_ids is not None:
                    kwargs["position_ids"] = position_ids
                outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]
            
            
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers[f"model.thinker.model.layers.{i}.{name}"] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            kwargs = {}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            outs[j] = layer(inps[j].unsqueeze(0), **kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if hasattr(config, "use_cache"):
        config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def qwen3ASR_eval(model, splits, dev, batch_size=8):
    print('Evaluating ...')

    decoder = model.model.thinker.model
    config = decoder.config
    
    use_cache = getattr(config, "use_cache", False)
    if hasattr(config, "use_cache"):
        config.use_cache = False

    wer_metric = evaluate_lib.load("wer")
    
    for split_name in splits:
        print(f"\n[{split_name}]")
        samples = load_librispeech_samples(split_name)

        print(f"  Running inference (batch_size={batch_size}) ...")
        result = evaluate_split(model, samples, wer_metric, batch_size, dev)

        print(f'WER ={result["wer"] * 100:.2f}%')
    

    if hasattr(config, "use_cache"):
        config.use_cache = use_cache

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )

    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )

    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sym', action='store_true')
    parser.add_argument('--true-sequential', action='store_true')
    parser.add_argument('--groupsize', type=int, default=-1)
    parser.add_argument('--act-order', action='store_true')
    parser.add_argument('--static-groups', action='store_true')
    
    args = parser.parse_args()

    model = get_qwen3(args.model)
    model.eval()

    dataloader = build_cali_dataloader(
            n_per_split=64,       # 총 128개
            seed=args.seed,
        )
    
    if args.wbits < 16:
        tick = time.time()
        quantizers = qwen3ASR_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    splits = ["test-clean", "test-other"]
    qwen3ASR_eval(model, splits, DEV)
