# AD-RD_no_OCBE

## dataflow
input -> E1(output = feature_a) -> E2(output = fearure_b) -> E3(output = feature_c) -> D3(output = feature_a) -> D2(output = feature_b) -> D1(output = fearure_c)

E1 vs D1 -> S1, E2 vs D2 -> S2, E3 vs D3 -> S3

## datatype
E1: [16, 256, 64, 64] -> [16, 256, 64, 64]   # stride = 1

E2: [16, 256, 64, 64] -> [16, 512, 32, 32]

E3: [16, 512, 32, 32] -> [16, 1024, 16, 16]

D3: [16, 1024, 16, 16] -> [16, 1024, 16, 16] # stride = 1

D2: [16, 1024, 16, 16] -> [16, 512, 32, 32]

D1: [16, 512, 32, 32]  -> [16, 256, 64, 64]

## main_swallowing.py
* bnすなわちOCBEに関わる部分の無効化
* D3への入力をOCBEの出力ではなくE3の出力に変更


```
encoder, bn = wide_resnet50_2(pretrained=True)
    -> encoder, _ = wide_resnet50_2(pretrained=True)
```

```
bn = bn.to(device)
    -> none
```

```
optimizer = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    -> optimizer = torch.optim.Adam(list(decoder.parameters()), lr=learning_rate, betas=(0.5, 0.999))
```

```
bn.train()
    -> none
```

```
inputs = encoder(img)
    -> inputs, feature_c = encoder(img) 
    # encoder(img)のreturnは[feature_a, feature_b, feature_c], feature_c
    # すなわちE1, E2, E3の出力のリスト（D1, D2, D3との比較用）とE3の出力（D3への入力用）
```

```
outputs = decoder(bn(inputs))
    -> outputs = decoder(feature_c)
```

```
auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device) 
    -> auroc_px, auroc_sp, aupro_px = evaluation(encoder, decoder, test_dataloader, device)
```

```
torch.save({'bn': bn.state_dict(), 'decoder': decoder.state_dict()}, ckp_path)
    -> torch.save({'decoder': decoder.state_dict()}, ckp_path)
```