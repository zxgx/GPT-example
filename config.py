data_path = 'toy.json'

BATCH_SIZE = 4
NUM_EPOCHS = 60
SEQ_LEN = 1024
NUM_CHUNKS = 1
TENSOR_SHAPE = (1, 1024, 1600)

NUM_MICRO_BATCHES = 4

NUM_LAYER = 48
HIDDEN_DIM = 1600
ATTENTION_HEAD = 32


parallel = dict(
    pipeline=4,
    # tensor=dict(mode='1d', size=2)
)

# fp16 = dict(
#     mode=AMP_TYPE.NAIVE
# )

if NUM_CHUNKS > 1:
    model = dict(
        num_chunks=NUM_CHUNKS,
    )
