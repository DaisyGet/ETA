import torch
from tqdm import tqdm
from .util import norm_logits, sample


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
@torch.no_grad()
def contrastive_sampling(x_1: torch.Tensor,
                         mask: torch.Tensor,
                         model: torch.nn.Module, model_s: torch.nn.Module, N: int, a: float = 0.5,
                            temperature: float = 1, top_k: int = 0., top_p: float = 0., eos_token_id: int = 2):
    n = len(x_1)
    T = len(x_1) + N
    past_key_values_1 = None
    past_key_values_2 = None

    model_dim = model.model_dim

    encoder = model.encoder.to(torch_device)
    decoder = model.decoder.to(torch_device)
    lm_head = model.lm_head.to(torch_device)

    encoder_s = model_s.encoder.to(torch_device)
    decoder_s = model_s.decoder.to(torch_device)
    lm_head_s = model_s.lm_head.to(torch_device)

    hidden_states = encoder(x_1, mask)[0]
    hidden_states_s = encoder_s(x_1, mask)[0]

    decoder_input_ids = torch.tensor([[0]]).to(torch_device)
    decoder_input_ids_s = torch.tensor([[0]]).to(torch_device)

    while n < T:
        # outputs = model(x)
        if past_key_values_1:
            last_ids_1 = decoder_input_ids[:, -1]
            last_ids_2 = decoder_input_ids_s[:, -1]
            if last_ids_1.dim() == 1:
                last_ids_1 = torch.unsqueeze(last_ids_1, 0)
            if last_ids_2.dim() == 1:
                last_ids_2 = torch.unsqueeze(last_ids_2, 0)
            decoder_outputs = decoder(last_ids_1, past_key_values=past_key_values_1, use_cache=True,
                                      encoder_hidden_states=hidden_states)
            decoder_outputs_s = decoder_s(last_ids_2, past_key_values=past_key_values_2, use_cache=True,
                                      encoder_hidden_states=hidden_states_s)
            # outputs_1 = model(last_ids_1, past_key_values=past_key_values_1, use_cache=True)
            # outputs_2 = model_s(last_ids_2, past_key_values=past_key_values_2, use_cache=True)
        else:
            # outputs_1 = model(x_1)
            # outputs_2 = model_s(x_2)
            decoder_outputs = decoder(decoder_input_ids, encoder_hidden_states=hidden_states)
            decoder_outputs_s = decoder_s(decoder_input_ids_s, encoder_hidden_states=hidden_states)

        sequence_output_1 = decoder_outputs[0]
        sequence_output_2 = decoder_outputs_s[0]
        sequence_output_1 = sequence_output_1 * (model_dim ** -0.5)
        sequence_output_2 = sequence_output_2 * (model_dim ** -0.5)

        logits_1 = lm_head(sequence_output_1)
        logits_2 = lm_head_s(sequence_output_2)
        contrastive_logits = (1+a)*logits_1[::, -1, :] - a * logits_2[::, -1, :]

        contrastive_logits = norm_logits(contrastive_logits, temperature, top_k, top_p)
        # last_p_2 = norm_logits(outputs_2.logits[::, -1, :], temperature, top_k, top_p)

        past_key_values_1 = decoder_outputs.past_key_values
        past_key_values_2 = decoder_outputs_s.past_key_values
        # last_p = (1+0.5)*last_p_1 - 0.5 * last_p_2
        idx_next = sample(contrastive_logits)
        # print('id_next', idx_next)
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        decoder_input_ids_s = torch.cat((decoder_input_ids_s, idx_next), dim=1)

        n += 1
        if idx_next == eos_token_id:
            break
        if eos_token_id in decoder_input_ids[0].tolist():
            # print('in')
            break

    return decoder_input_ids
