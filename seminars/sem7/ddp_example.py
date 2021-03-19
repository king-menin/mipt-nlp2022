#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoModelForCausalLM
from torch.optim import Adam
from demo_dataset import DemoDataset


NUM_EPOCHS = 3


def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    args = parser.parse_args()

    # initialize your model
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    # send your model to GPU
    model.cuda()
    
    optimizer = Adam(model.parameters(), lr=1e-5)

    # initialize your dataset
    dataset = DemoDataset()

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
            batch = batch.cuda()
            
            # forward pass
            loss, _, _ = model(batch, labels=batch)

            # backward pass
            loss.backward()
            optimizer.step()
            if step % 40 == 0:
                print('Loss %.3f' % loss)

if __name__ == '__main__':
    main()