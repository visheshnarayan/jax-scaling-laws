from model.transformer import TransformerConfig

MODEL_FAMILY = {                                                                                                                                               
      "3M":   TransformerConfig(n_layers=2,  d_model=64,  n_heads=2,  d_ff=256),
      "6M":   TransformerConfig(n_layers=4,  d_model=96,  n_heads=3,  d_ff=384),
      "10M":  TransformerConfig(n_layers=4,  d_model=128, n_heads=4,  d_ff=512),
      "17M":  TransformerConfig(n_layers=6,  d_model=192, n_heads=6,  d_ff=768),
      "30M":  TransformerConfig(n_layers=8,  d_model=256, n_heads=8,  d_ff=1024),
      "50M":  TransformerConfig(n_layers=8,  d_model=384, n_heads=6,  d_ff=1536),
      "80M":  TransformerConfig(n_layers=10, d_model=512, n_heads=8,  d_ff=2048),
      "110M": TransformerConfig(n_layers=12, d_model=640, n_heads=10, d_ff=2560),
}

HELD_OUT = {
    "150M": TransformerConfig(n_layers=12, d_model=768, n_heads=12, d_ff=3072),
}                                                                                                                                                              

if __name__ == "__main__":
    from model.init import count_params
    
    for name, cfg in {**MODEL_FAMILY, **HELD_OUT}.items():
        n = count_params(cfg)
        print(f"{name:>5s}: {n:>12,} params | layers={cfg.n_layers} d={cfg.d_model} heads={cfg.n_heads}")