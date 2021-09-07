import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, gzip, pickle
from src.data.vocabulary import GlsTextVocab
import random


class SignRegTranDataset(Dataset):
    def __init__(self, opts, gls_text_vocab, phrase, sample=True, DEBUG=False):
        super(SignRegTranDataset, self).__init__()
        self.data_path = opts.data_path
        self.opts = opts
        self.phrase = phrase
        self.sample = sample
        self.gls_text_vocab = gls_text_vocab
        self.max_sgn_len = 300
        self.text_bos = self.gls_text_vocab.token2idx("<s>")
        self.text_eos = self.gls_text_vocab.token2idx("</s>")

        self.phoenix_dataset = self.load_data_list()

        self.data_dict = self.phoenix_dataset[phrase]
        # if DEBUG == True:
        #     self.data_dict = self.data_dict[:101]

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        cur_info = self.data_dict[idx]
        video_name = cur_info["name"]
        # print(cur_info["sign_feature"].shape, cur_info.keys())
        # exit()
        sign_feature = self._temporal_sample(cur_info["sign_feature"])
        sign_len = sign_feature.size(0)
        gloss = cur_info["gloss"]
        gloss_len = len(gloss)
        text = cur_info["text"]
        text_len = len(text)
        sample = {"name": video_name, "sign_feature": sign_feature, "sign_len": sign_len,
                  "gloss": gloss, "gloss_len": gloss_len, "text": text, "text_len":text_len}
        return sample

    def _temporal_sample(self, sgn_feature):
        """ temporal sample for sign features.

        :param sgn_feature: [len, 1024]
        :return:
        """
        # print("sgn_feature: ", sgn_feature.shape)

        if self.phrase == 'train' and self.sample:
            # first, Randomly repeat 20%. Second, Randomly delete 20%
            ids = list(range(sgn_feature.size(0)))
            add_idx = random.sample(ids, int(0.2 * len(ids)))
            ids.extend(add_idx)
            ids.sort()
            ids = random.sample(ids, int(0.8 * len(ids)))
            ids.sort()
            if len(ids) > self.max_sgn_len:
                ids = random.sample(ids, self.max_sgn_len)
                ids.sort()
            sgn_feature = sgn_feature.index_select(0, torch.LongTensor(ids))
        return sgn_feature


    def load_data_list(self):
        phoenix_dataset = {}
        for task in ["train", "dev", "test"]:
            if task != self.phrase:
                continue
            corpus_path = os.path.join(self.data_path, "phoenix14t.pami0.{}".format(self.phrase))
            with gzip.open(corpus_path, "rb") as f:
                corpus = pickle.load(f)
                print("The number of {} datasets is {}".format(self.phrase, len(corpus)))
            videonames, glosses, texts, sign_features = [], [], [], []
            for data in corpus:
                videonames.append(data["name"])
                glosses.append(data["gloss"])
                texts.append(data["text"])
                sign_features.append(data["sign"])
            data_infos = []
            for i in range(len(videonames)):
                tmp_info = {
                    "name" : videonames[i],
                    # "sign_feature" : torch.LongTensor(sign_features[i]).squeeze(),
                    "sign_feature" : sign_features[i],
                    "gloss_sent" : glosses[i],
                    "gloss": self.gls_text_vocab.gloss_sent_to_ids(glosses[i]),
                    "text_sent": texts[i],
                    "text": self.gls_text_vocab.text_sent_to_ids(texts[i])
                }
                data_infos.append(tmp_info)
            phoenix_dataset[task] = data_infos
        return phoenix_dataset


    def collate_fn(self, batch):
        sign_len = [x["sign_len"] for x in batch]
        gloss_len = [x["gloss_len"] for x in batch]
        text_len = [x["text_len"] for x in batch]

        batch_sign_feat = torch.zeros(len(batch), max(sign_len), self.opts.input_size)  # padding with zeros
        batch_gloss = torch.ones(len(batch), max(gloss_len)).long() * self.gls_text_vocab.gloss2idx("<pad>")
        batch_text_inp = torch.ones(len(batch), max(text_len) + 1).long() * self.gls_text_vocab.token2idx("<pad>")
        batch_text_trg = torch.ones(len(batch), max(text_len) + 1).long() * self.gls_text_vocab.token2idx("<pad>")

        vname = []
        for i, bat in enumerate(batch):
            batch_sign_feat[i, :sign_len[i]] = bat["sign_feature"]
            batch_gloss[i, :gloss_len[i]] = torch.LongTensor(bat["gloss"])
            batch_text_inp[i, :text_len[i]+1] = torch.LongTensor([self.text_bos] + bat["text"])
            batch_text_trg[i, :text_len[i]+1] = torch.LongTensor(bat["text"] + [self.text_eos])
            vname.append(bat["name"])

        sign_len = torch.LongTensor(sign_len)
        gloss_len = torch.LongTensor(gloss_len)
        text_len = torch.LongTensor(text_len) + 1

        return {"name": vname, "sign_feature": batch_sign_feat, "sign_len": sign_len,
                "gloss": batch_gloss, "gloss_len": gloss_len,
                "text_inp": batch_text_inp, "text_trg": batch_text_trg, "text_len": text_len}


if __name__ == "__main__":
    class Config():
        data_path = "data_bin/PHOENIX2014T"

    opts = Config()
    dataset = SignRegTranDataset(opts, phrase="train")
    # for i, data in enumerate(dataset):
    #     if i > 5:
    #         break
    #     print(data["sign_feature"].shape)
    #     print(data["gloss"])
    #     print(data["text"])

    train_iter = DataLoader(dataset,
                            batch_size=5,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=dataset.collate_fn,
                            drop_last=True)

    for data in train_iter:
        print(data.keys())
        print(data["sign_feature"].shape)
        print("gloss: ", data["gloss"])
        print("text_inp: ", data["text_inp"])
        print("text_trg: ", data["text_trg"])
        exit()

    # import gzip, pickle
    # path = "data_bin/PHOENIX2014T/phoenix14t.pami0.test"
    # with gzip.open(path, "rb") as f:
    #     ob = pickle.load(f)
    # print("data number: ", len(ob))
    # for i, data in enumerate(ob):
    #     if i > 5:
    #         break
    #     print(data["sign"].shape)
    #     print(data["name"])
    #     print(data["signer"])
    #     print(data["gloss"])
    #     print(data["text"])
