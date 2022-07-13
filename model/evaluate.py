import math
import random
import torch
import time

from model.model import Encoder, Decoder
from typing import List
from model.vocabulary import EOS_token, SOS_token, tensor_from_sentence


def evaluate(encoder: Encoder, decoder: Decoder,
             sequence: torch.Tensor,
             device: torch.device,
             max_length: int) -> List[int]:
    with torch.no_grad():
        sequence_length = sequence.size()[0]

        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(sequence_length):
            encoder_output, encoder_hidden = encoder(sequence[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []

        for _ in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(topi.item())
            
            decoder_input = topi.squeeze().detach()
        
        return decoded_words