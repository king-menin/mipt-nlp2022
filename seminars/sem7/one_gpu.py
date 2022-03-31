#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from transformers import AutoModelForCausalLM
from torch.optim import Adam
from tokenize_data import tokenize_and_load, add_tokenize_data_args


NUM_EPOCHS = 3


def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    parser = add_tokenize_data_args(parser)
    args = parser.parse_args()

    # initialize your model
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    # send your model to GPU
    model.cuda()
    
    optimizer = Adam(model.parameters(), lr=1e-5)

    # initialize your dataset
    texts, _ = tokenize_and_load(**vars(args))
    dataset = TensorDataset(torch.LongTensor(texts))

    # initialize Sampler
    sampler = SequentialSampler(dataset)

    # initialize the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size
    )

    # start your training!
    for epoch in range(NUM_EPOCHS):
        # put model in train mode
        model.train()

        for step, batch in enumerate(dataloader):
            # send batch to device
            batch = batch[0] 
            batch = batch.cuda()
            
            # forward pass
            out = model(batch, labels=batch)
            loss = out["loss"]
            # backward pass
            loss.backward()
            optimizer.step()
            if step % 40 == 0:
                print('Loss %.3f' % loss)

if __name__ == '__main__':
    main()
