import torch

from model.model import Decoder
from model.vocabulary import Vocabulary, EOS_token, SOS_token

def generate(decoder: Decoder, vocabulary: Vocabulary, latent_dim: int, max_length: int, device: torch.device) -> str:
    with torch.no_grad():
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = torch.zeros(1, 1, latent_dim, device=device)

        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                word = vocabulary.index2word[topi.item()]
                decoded_words.append(word)
            
            decoder_input = topi.squeeze().detach()
        
        return ' '.join(decoded_words)