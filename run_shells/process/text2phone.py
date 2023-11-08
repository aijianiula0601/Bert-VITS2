from text import tokenizer
from torchtext.vocab import Vocab

from text.symbols import special_symbols, UNK_ID
from typing import List
from torchtext.vocab import build_vocab_from_iterator


def text2phone(text, text_cleaners=[['phonemize_text'], ['add_spaces']], language='en-us'):
    text_norm = [text]

    for cleaners in text_cleaners:
        print(f"Cleaning with {cleaners} ...")
        if cleaners[0] == "phonemize_text":
            text_norm = tokenizer(text_norm, Vocab, cleaners, language=language)
        else:
            for idx, text in enumerate(text_norm):
                temp = tokenizer(text, Vocab, cleaners, language=language)
                text_norm[idx] = temp

    return text_norm[0]


def yield_tokens(cleaned_text: List[str]):
    for text in cleaned_text:
        yield text.split()


def save_vocab(vocab, vocab_file: str):
    """Save vocabulary as token index pairs in a text file, sorted by the indices
    Args:
        vocab (torchtext.vocab.Vocab): Vocabulary object
        vocab_file (str): Path to vocabulary file
    """
    with open(vocab_file, "w") as f:
        for token, index in sorted(vocab.get_stoi().items(), key=lambda kv: kv[1]):
            f.write(f"{token}\t{index}\n")


def save_all_vocab(phone_list: list, vocab_file: str):
    vocab = build_vocab_from_iterator(yield_tokens(phone_list), specials=special_symbols)
    vocab.set_default_index(UNK_ID)

    save_vocab(vocab, vocab_file)

    print(f"save vocab to:{vocab_file}")


def load_vocab(vocab_file: str):
    """Load vocabulary from text file
    Args:
        vocab_file (str): Path to vocabulary file
    Returns:
        torchtext.vocab.Vocab: Vocabulary object
    """
    from torchtext.vocab import vocab as transform_vocab
    from text.symbols import UNK_ID, special_symbols

    vocab = {}
    with open(vocab_file, "r") as f:
        for line in f:
            token, index = line.split()
            vocab[token] = int(index)
    vocab = transform_vocab(vocab, specials=special_symbols)
    vocab.set_default_index(UNK_ID)
    return vocab


if __name__ == '__main__':
    text = 'Printing, in the only sense with which we are'
    phone = text2phone(text, language='en')

    print("----phone:", phone)

    save_all_vocab([phone], '/tmp/test_vocab.txt')

    vocab = load_vocab('/tmp/test_vocab.txt')
    print("----vocab:", vocab)


    indices = vocab.lookup_indices(phone.split())
    print("---indices:", indices)
