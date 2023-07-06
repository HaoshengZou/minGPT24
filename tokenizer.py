from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.decoders import Decoder
from transformers import PreTrainedTokenizerFast

all_chars = "0123456789[,]:+-*/= "
itoc = {i:c for i, c in enumerate(all_chars)}
ctoi = {c:i for i, c in enumerate(all_chars)}


class DummySplit:
    def dummy_split(self, i, norm_str):
        ans = [norm_str[j: j + 1] for j in range(len(str(norm_str)))]
        return ans

    def pre_tokenize(self, pretok):
        pretok.split(self.dummy_split)


class DummyJoin:
    def decode_chain(self, tokens):
        return [f"{t}" for t in tokens]


def get_TokenizerV0(for_trl=False):
    tok = Tokenizer(WordLevel(ctoi, unk_token="[UNK]"))
    if for_trl:
        tok = PreTrainedTokenizerFast(tokenizer_object=tok)
        tok._tokenizer.pre_tokenizer = PreTokenizer.custom(DummySplit())
        tok._tokenizer.decoder = Decoder.custom(DummyJoin())
    else:
        tok.pre_tokenizer = PreTokenizer.custom(DummySplit())
        tok.decoder = Decoder.custom(DummyJoin())
    return tok


if __name__ == '__main__':
    ##### v0 tokenizer #####
    tok = get_TokenizerV0(for_trl=True)

    res = tok.encode('[4, 8, 9, 3]: 8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  ')
    print(res)
    print(tok.decode(res))

    exit()
    inputs = [
        '[4, 8, 9, 3]: 8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  ',
        '[4, 8, 9, 3]: 8 + 4 = 12, 9 + 3 = 12, 12 + 12 = 24  '
    ]
    res = tok.encode(inputs)
    print(inputs)
    from pprint import pprint
    pprint(res)
    print()
    print(len(res))
