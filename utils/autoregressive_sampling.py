import torch
from tqdm import tqdm
from .util import norm_logits, sample

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
@torch.no_grad()
def autoregressive_sampling(x: torch.Tensor,
                            mask: torch.Tensor,
                            model: torch.nn.Module, N: int,
                            temperature: float = 1, top_k: int = 0., top_p: float = 0., eos_token_id: int = 2):
    n = len(x)
    T = len(x) + N

    past_key_values = None

    model_dim = model.model_dim

    encoder = model.encoder.to(torch_device)
    decoder = model.decoder.to(torch_device)
    lm_head = model.lm_head.to(torch_device)

    encoder_outputs = encoder(x, mask)
    hidden_states = encoder_outputs[0]
    decoder_input_ids = torch.tensor([[0]]).to(torch_device)

    # print('eos', eos_token_id)
    while n < T:
        if past_key_values:
            last_ids = decoder_input_ids[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            decoder_outputs = decoder(last_ids, past_key_values=past_key_values, use_cache=True, encoder_hidden_states=hidden_states)
        else:
            decoder_outputs = decoder(decoder_input_ids, encoder_hidden_states=hidden_states)
        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (model_dim ** -0.5)

        logits = lm_head(sequence_output)
        last_p = norm_logits(logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = decoder_outputs.past_key_values
        idx_next = sample(last_p)
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        n += 1
        # print('decoder_input_ids')
        # print(decoder_input_ids)
        if idx_next == eos_token_id:
            break
        if eos_token_id in decoder_input_ids[0].tolist():
            break
    return decoder_input_ids