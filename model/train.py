import itertools
import math
import random
import torch
import time

from model.model import Encoder, Decoder
from torch import optim, nn
from typing import Iterable
from model.vocabulary import EOS_token, SOS_token, tensor_from_sentence
from numbers import Number
from util.plot import show_plot


teacher_forcing_ratio = 0.5

def train_one_iteration(
        encoder: Encoder, decoder: Decoder,
        sequence: torch.Tensor,
        encoder_optimizer, decoder_optimizer,
        criterion,
        device: torch.device,
        max_length: int) -> float:
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    sequence_length = sequence.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for t in range(min(sequence_length, max_length)):
        encoder_output, encoder_hidden = encoder(sequence[t], encoder_hidden)
        encoder_outputs[t] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)
    
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        for t in range(sequence_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, sequence[t])
            decoder_input = sequence[t] 
    else:
        for t in range(sequence_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, sequence[t])
            if decoder_input.item() == EOS_token:
                break
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / sequence_length


def as_minutes(seconds: Number) -> str:
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)


def time_since(since, percent) -> str:
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return as_minutes(s)


def train(encoder: Encoder, decoder: Decoder,
          sequences: Iterable[torch.Tensor],
          device: torch.device,
          max_length: int,
          print_every=1000, plot_every=100,
          learning_rate=0.01,
          iterations=100) -> None:
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    print('time elapsed', 'iteration', 'loss', sep='\t')
    for iteration, sequence in enumerate(itertools.islice(sequences, iterations), start=1):
        loss = train_one_iteration(encoder, decoder, sequence, encoder_optimizer, decoder_optimizer, criterion, device, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(time_since(start, iteration / iterations), iteration, print_loss_avg, sep='\t')
        
        if iteration % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    show_plot(plot_losses)