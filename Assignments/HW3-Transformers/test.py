class GenericTransformer(DummyTransformer): 

    def __init__(self, config): 

        super().__init__(config, TransformerBlock, Embedding) 

        self.block_size = config.block_size # Maximum Number of Tokens which can be encoded at once 

        self.vocab_size = config.vocab_size 

 

    def get_attention_mask(self, num_tokens): 

        """ 

        Dummy For now, we will see how we use this later! 

        """ 

        B = num_tokens.shape[0] 

        return torch.ones((B, self.block_size, self.block_size))[:, :num_tokens.max().item(), :num_tokens.max().item()] 

 

    def forward(self, idx, targets=None, hidden_cache=None, return_hidden=False): 

        """ 

        :param idx: int Tensor of shape (B,T) 

        :param hidden_cache: float Tensor of shape (B,P_T,n_embd) 

        :param targets: int Tensor of shape (B,T_T) 

        :param return_hidden: bool 

        (if return_hidden = None) 

        :returns x: float Tensor of shape (B,T,n_embd) 

        (else) 

        :returns logits: float Tensor of shape (B, T, vocab_size) 

        :returns loss: float Tensor of shape (B) or None 

        """ 

        num_tokens = (idx != -1).type(torch.int).sum(dim=1) 
        if hidden_cache is not None: 
          num_tokens = num_tokens + hidden_cache.shape[1] 
        idx = idx.masked_fill(idx == -1, int(0)).type(torch.int)[:, :num_tokens.max().item()] 
        if targets is not None: 
          targets = targets[:, :num_tokens.max().item()] 
        attention_mask = self.get_attention_mask(num_tokens) 

        # Embedding 
        embeddings = self.transformer['embedding'](idx)  # (B, T, n_embd) 

        if hidden_cache is not None: 
            embeddings = torch.cat([hidden_cache, embeddings], dim=1)  # Concatenate in the token dimension 

        # Transformer blocks 
        x = embeddings 
        for block in self.transformer['h']: 
            x = block(x, attention_mask) 

        # Final layer normalization 
        x = self.transformer['ln_f'](x) 

        # Project to vocab size 
        logits = self.lm_head(x)  # (B, T, vocab_size) 