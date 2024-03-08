import pickle
import tqdm
from collections import Counter


class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Conuter object holding the frequencies of tokens
            in the data used to build the Vocab
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers
        itos: A list of token strings indexd by their numerical identifiers.        
    """

    # max_size=None 이라 최대 크기에 제한이 없다.
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Conuter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vector_cache: direcory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of speicial tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]
        
        # None이 아니면 특별 토큰들의 개수를 추가하여 최대 어휘 크기를 설정한다.
        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, the alphabetically
        words_and_frequencies = sorted(counter.item(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True
    
    def __len__(self):
        return len(self.itos)
    
    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)
    
    # 문장을 시퀀스로 변환한다. 이때, 문장, 시퀀스 길이, 문장의 시작과 끝에 특별 토큰 추가 옵션을 제공
    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # 시퀀스를 문장으로 변환
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocba_path: str) -> 'Vocab': # 단어 집합을 파일에서 로드
        with open(vocba_path, "rb") as f:
            return pickle.load(f)
        
    def save_vocab(self, vocab_path): # 단어 집합을 파일에 저장
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# Building Vocab with text files
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in tqdm.tqdm(texts):
            if isinstance(line, list):
                words = line
            
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index] # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq
    
    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]
        
        return " ".join(words) if join else words
    
    # 정적메서드는 클래스에 속하지만, 인스턴스에 종속되지 않는 메서드,
    # 즉, 인스턴스의 속성이나 메서드에 접근하지 않고 클래스의 네임스페이스에 정의된 메서드
    @staticmethod
    def load_vocab(vocba_path: str) -> WordVocab:
        with open(vocba_path, "rb") as f:
            return pickle.load(f)


def build():
    # 명령행 인자를 파싱하기 위한 클래스를 초기화 하는 것
    """여기서 argparse는 명령행 인자를 해석하고 처리하는 기능을 제공하는 라이브러리입니다. 
    ArgumentParser 클래스는 프로그램에 필요한 명령행 인자를 정의하고, 
    사용자가 프로그램을 실행할 때 전달할 수 있는 인자를 파싱하고 처리할 수 있는 인터페이스를 제공합니다.
    """
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)  

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)