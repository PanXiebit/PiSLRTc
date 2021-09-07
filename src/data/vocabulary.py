import os
import gzip
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SIL_TOKEN = "<blank>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


class GlsTextVocab(object):
    def __init__(self, opts):
        self.data_path = opts.data_path
        self.gloss2idx_dict, self.idx2gloss_dict, self.token2idx_dict, self.idx2token_dict = \
            self.get_and_save_vocabulary(
                train_path=os.path.join(self.data_path, "phoenix14t.pami0.train"),
                gloss_vocab_path=os.path.join(self.data_path, "gloss_vocab.txt"),
                text_vocab_path=os.path.join(self.data_path, "text_vocab.txt"))

    def gloss_vocab_len(self):
        return len(self.gloss2idx_dict)

    def text_vocab_len(self):
        return len(self.token2idx_dict)

    def gloss2idx(self, gloss):
        return self.gloss2idx_dict[gloss]

    def idx2gloss(self, idx):
        return self.idx2gloss_dict[idx]

    def gloss_sent_to_ids(self, gloss_sent):
        gloss_sent = gloss_sent.split(' ')
        s = []
        for gloss in gloss_sent:
            if gloss in self.gloss2idx_dict:
                s.append(self.gloss2idx(gloss))
            else:
                s.append(self.gloss2idx("<unk>"))
        return s

    def gloss_lis_to_sentences(self, idx_list) -> str:
        gloss_sequences = []
        for idx in idx_list:
            gloss_sequences.append(self.idx2gloss(idx))
        return " ".join(gloss_sequences)

    def token2idx(self, token):
        return self.token2idx_dict[token]

    def idx2token(self, idx):
        return self.idx2token_dict[idx]

    def text_sent_to_ids(self, token_sent):
        token_sent = token_sent.split(' ')
        s = []
        for token in token_sent:
            if token in self.token2idx_dict:
                s.append(self.token2idx(token))
            else:
                s.append(self.token2idx("<unk>"))
        return s

    def text_list_to_sentences(self, idx_list) -> str:
        text_sequences = []
        for idx in idx_list:
            if idx == self.token2idx(EOS_TOKEN):
                break
            text_sequences.append(self.idx2token(idx))
        return " ".join(text_sequences)

    @staticmethod
    def get_and_save_vocabulary(train_path, gloss_vocab_path, text_vocab_path):
        if not os.path.exists(gloss_vocab_path):
            with gzip.open(train_path, "rb") as f:
                train_data = pickle.load(f)
            gloss_tokens = defaultdict(int)
            text_tokens = defaultdict(int)
            for content in tqdm(train_data):
                gloss_sent = content["gloss"]
                text_sent = content["text"]
                for gloss in gloss_sent.strip().split():
                    gloss_tokens[gloss] += 1
                for token in text_sent.strip().split():
                    text_tokens[token] += 1

            gloss_tokens_sort = sorted(gloss_tokens.items(), key=lambda item: item[1], reverse=True)
            text_tokens_sort = sorted(text_tokens.items(), key=lambda item: item[1], reverse=True)

            gloss_special_token = [SIL_TOKEN, UNK_TOKEN, PAD_TOKEN]
            text_special_token = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

            with open(gloss_vocab_path, "w") as f:
                for gloss in gloss_special_token:
                    if gloss not in gloss_tokens:
                        f.write(gloss + "\t" + str(1) + "\n")
                for gloss, cnt in gloss_tokens_sort:
                    f.write(gloss + "\t" + str(cnt) + "\n")

            with open(text_vocab_path, "w") as f:
                for word in text_special_token:
                    if word not in text_tokens:
                        f.write(word + "\t" + str(1) + "\n")
                for word, cnt in text_tokens_sort:
                    f.write(word + "\t" + str(cnt) + "\n")

        gloss2idx_dict = {}
        idx2gloss_dict = {}
        with open(gloss_vocab_path, "r") as f:
            for line in f:
                content = line.strip().split("\t")
                idx = len(gloss2idx_dict)
                gloss2idx_dict[content[0]] = idx
                idx2gloss_dict[idx] = content[0]

        token2idx_dict = {}
        idx2token_dict = {}
        with open(text_vocab_path, "r") as f:
            for line in f:
                content = line.strip().split()
                idx = len(token2idx_dict)
                token2idx_dict[content[0]] = idx
                idx2token_dict[idx] = content[0]

        print("gloss_vocab size: ", len(gloss2idx_dict))
        print("text_vocab size: ", len(token2idx_dict))
        return gloss2idx_dict, idx2gloss_dict, token2idx_dict, idx2token_dict


if __name__ == "__main__":
    train_path = "data_bin/PHOENIX2014T/phoenix14t.pami0.train"
    gloss_vocab_path = "data_bin/PHOENIX2014T/gloss_vocab.txt"
    text_vocab_path = "data_bin/PHOENIX2014T/text_vocab.txt"
    GlsTextVocab.get_and_save_vocabulary(train_path, gloss_vocab_path, text_vocab_path)
