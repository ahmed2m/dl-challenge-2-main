import torch
from torch import nn

# https://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://d2l.ai/chapter_recurrent-modern/seq2seq.html#decoder
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb


class Encoder(nn.Module):
    def __init__(
        self, input_size, vocab_size, embedding_size, hidden_size
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = nn.LSTM(
            embedding_size,
            hidden_size,
            batch_first=True,
        )

    def forward(self, x):
        x = self.embedding(x)
        out, (hidden, current) = self.encoder(x)

        return hidden, current


class Decoder(nn.Module):
    def __init__(
        self, input_size, vocab_size, embedding_size, hidden_size
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.decoder = nn.LSTM(
            embedding_size,
            hidden_size,
            # batch_first=True, # for some reason batch_first for encoder doesn't apply to hidden
        )
        self.ll = nn.Linear(hidden_size, vocab_size)
        self.decoder_out = nn.LogSoftmax(1)

    def forward(self, x, hidden, current):
        x = x.unsqueeze(0)
        x = self.embedding(x)
        out, (hidden, current) = self.decoder(x, (hidden, current))
        out = self.ll(out)
        # out = self.decoder_out(out.squeeze(0))
        return out, hidden, current


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input, target, is_training=True):
        target_size = target.shape[1]
        vocab_size = self.decoder.vocab_size
        batch_size = input.shape[0]

        hidden, current = self.encoder(input)
        # out = self.decoder(target, hidden, current)

        outputs = torch.zeros(batch_size, target_size, vocab_size).to(
            self.device
        )

        target_input = target[:, 0]

        for i in range(target_size):
            out, hidden, current = self.decoder(target_input, hidden, current)
            # out = out.view(batch_size, vocab_size)
            outputs[:, i] = out.squeeze(0)
            if is_training:
                target_input = target[:, i]
            else:
                prediction = out.squeeze(0).argmax(1)
                target_input = prediction
        return outputs
