import torch.nn as nn

from .bert import BERT

class BERTLM(nn.Module):
    '''
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    '''

    # :기호는 타입을 명시하는데 사용한다.
    def __init__(self, bert: BERT, vocab_size):
        '''
        :param bert: BERT model which should be trained
        :param vocab_size: total bocab size for masked_lm
        '''

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label) ## ???segment_label
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classfication model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1) # ????

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0])) # 다음 문장 예측을 위해 CLS 토큰의 특성만을 사용한다는 것을 의미
    

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classfication problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocabs_size):
        """
        :param hidden: ouput size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocabs_size)
        self.softmax = nn.LogSoftmax(dim=-1) # 마지막 차원을 기준으로 소프트맥스를 계산하고 로그를 취한다.

    def forward(self, x):
        return self.softmax(self.linear(x))