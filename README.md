# CS Theory Papers Model

**Author:** Abhinav Madahar <abhinavmadahar@gmail.com>

This is a machine learning model which generates paper abstracts in the theory subfield of computer science.
I can't find a dataset of CS theory papers, but I can make a dataset of those papers' abstracts.

## How to use this model

First, use the `data/download.sh` script to download abstracts, and save the output to a text file.
Then download the BibTeX from [the ACL Anthology website](https://aclanthology.org/), specifying the one with abstracts.

## Todo

The todo is expected to get more sub-tasks as the project continues.

- [x] Get dataset of paper abstracts in CS theory
- [x] Select a model architecture to use
- [ ] Implement the model
- [ ] Fine-tune the model
- [ ] Publish results

## Model design

The model is a text generation model.
It generates a paper abstract in CS theory as the output.
The output text has no formatting, consisting solely of raw text.

To create the model, we first create a model with an encoder-decoder architecture with no attention.
This model learns to encapsulate abstracts into a fixed-length vector.
The encoder reads an abstract, and we initialize the decoder with its final state(s).
The decoder then tries to recreate the original abstract.
After this encoder-decoder model is decent, we can generate novel abstracts by passing new vectors to the decoder.

To train the encoder-decoder model, we first train it on general text, like tweets or Shakespeare's writing, to teach it English.
After that, we train it further on the abstracts.
It might be useful to first train it on more general abstracts, like abstracts across computer science, mathematics, physics, etc., so it learns how to discuss science.