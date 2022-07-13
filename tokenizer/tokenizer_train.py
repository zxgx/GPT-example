import os
from tokenizers import ByteLevelBPETokenizer


def train(files, vocab_size=30000, min_freq=2, special_tokens=None, save_path='bbpe.tok'):
    tok = ByteLevelBPETokenizer()
    tok.train(files, vocab_size, min_frequency=min_freq, special_tokens=special_tokens)
    tok.enable_truncation(max_length=1024)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    tok.save_model(str(save_path))


if __name__ == '__main__':
    train(['train.txt'], special_tokens=['<|endoftext|>'])
