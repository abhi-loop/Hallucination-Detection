def extract_sentence_embedding(hidden_states):
    """
    hidden_states: tuple[num_layers] of (1, seq_len, hidden_dim)
    Paper choice: middle layer, last token
    """
    num_layers = len(hidden_states)
    middle_layer = hidden_states[num_layers // 2]
    last_token_embedding = middle_layer[:, -1, :]
    return last_token_embedding.squeeze(0)