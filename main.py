import torch
import torch.nn as nn

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.core import global_context as gpc
import colossalai.utils as utils
from model_zoo.gpt import GPTLMLoss
from colossalai.context.parallel_mode import ParallelMode
from colossalai.utils.timer import MultiTimer
from colossalai.trainer import Trainer, hooks

from utils import slurm_init, WebtextDataset
from modules.gpt import build_pipeline_model


def main():
    disable_existing_loggers()
    slurm_init()
    logger = get_dist_logger()

    train_ds = WebtextDataset(gpc.config.data_path, seq_len=gpc.config.SEQ_LEN, cache_dir='gpt2_tokenizer')
    train_dataloader = utils.get_dataloader(train_ds, seed=42,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=True, shuffle=True, drop_last=True)

    vocab_size = train_ds.tokenizer.vocab_size
    if vocab_size % 2 == 1:
        vocab_size += 1

    padding_idx = train_ds.tokenizer.convert_tokens_to_ids(train_ds.tokenizer.pad_token)
    model = build_pipeline_model(gpc.config.NUM_LAYER, gpc.config.NUM_CHUNKS, vocab_size=vocab_size,
                                 max_seq_len=gpc.config.SEQ_LEN, padding_idx=padding_idx,
                                 hidden_dim=gpc.config.HIDDEN_DIM, num_head=gpc.config.ATTENTION_HEAD,
                                 checkpoint=True)

    if gpc.config.NUM_CHUNKS > 1 and not isinstance(model, nn.ModuleList):
        model = nn.ModuleList([model])

    criterion = GPTLMLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-2, )
    engine, train_dataloader, _, _ = colossalai.initialize(model,
                                                           optimizer,
                                                           criterion,
                                                           train_dataloader=train_dataloader)
    global_batch_size = gpc.config.BATCH_SIZE * \
                        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    timer = MultiTimer()
    trainer = Trainer(
        engine=engine,
        logger=logger,
        timer=timer
    )
    hook_list = [
        hooks.LossHook(),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(),
        hooks.LogMetricByStepHook(),
    ]

    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_interval=1,
        hooks=hook_list,
        display_progress=True,
        return_output_label=False,
    )


if __name__ == '__main__':
    main()
