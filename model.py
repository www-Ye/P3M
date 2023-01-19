from cProfile import label
import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from long_seq import process_long_input
import random

class DocREModel(nn.Module):
    def __init__(self, args, config, priors_l, priors_o, model, emb_size=768, block_size=64):
        super().__init__()
        self.args = args
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.priors_l = priors_l
        self.priors_o = priors_o
        self.w = ((1 - self.priors_o)/self.priors_o) ** 0.5
        self.rels = args.num_class-1

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, emb_size)

        self.r_emb = nn.Embedding(args.num_class, config.hidden_size)
        self.r = torch.LongTensor([x for x in range(args.num_class)]).to(0)
        self.re_linear = nn.Linear(config.hidden_size, emb_size)

        self.loss = self.SoftMax_norm

        self.emb_size = emb_size
        self.block_size = block_size

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

            if len(hts[i]) == 0:
                hss.append(torch.FloatTensor([]).to(sequence_output.device))
                tss.append(torch.FloatTensor([]).to(sequence_output.device))
                rss.append(torch.FloatTensor([]).to(sequence_output.device))
                continue
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def SoftMax_norm(self, pos, neg, la=10.):
        if len(pos) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = (- torch.log(torch.exp(la*pos) / (torch.exp(la*pos) + torch.exp(la*(neg)))))
        return loss.mean()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                eval=False,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))    # zs
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))    # zo
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

        emb = self.bilinear(bl)
        emb_norm = F.normalize(emb, p=2, dim=-1)

        r_emb = self.r_emb(self.r)
        r_emb = self.re_linear(r_emb)
        r_emb_norm = F.normalize(r_emb, p=2, dim=-1)

        logits = emb_norm.matmul(r_emb_norm.T)

        if labels is not None:

            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)

            if (not eval) and (self.args.aug != 'no_aug'):

                sequence_output1, attention1 = self.encode(input_ids, attention_mask)
                hs1, rs1, ts1 = self.get_hrt(sequence_output1, attention1, entity_pos, hts)

                hs1 = torch.tanh(self.head_extractor(torch.cat([hs1, rs1], dim=1)))    # zs
                ts1 = torch.tanh(self.tail_extractor(torch.cat([ts1, rs1], dim=1)))    # zo
                b1_1 = hs1.view(-1, self.emb_size // self.block_size, self.block_size)
                b2_1 = ts1.view(-1, self.emb_size // self.block_size, self.block_size)
                bl_1 = (b1_1.unsqueeze(3) * b2_1.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

                emb1 = self.bilinear(bl_1)
                emb1_norm = F.normalize(emb1, p=2, dim=1)
                logits1 = emb1_norm.matmul(r_emb_norm.T)

                all_emb_norm = torch.cat([emb_norm, emb1_norm], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)

                if self.args.aug == 'pos_aug':
                    logits = torch.cat([logits, logits1[labels[:, 0] != 1]], dim=0)
                    labels = torch.cat([labels, labels[labels[:, 0] != 1]], dim=0)
                elif self.args.aug == 'all_aug':
                    logits = torch.cat([logits, logits1], dim=0)
                    labels = torch.cat([labels, labels], dim=0)
                else:
                    print('error')
                    raise NameError
            else:
                all_emb_norm = emb_norm
                all_labels = labels

            if (not eval) and (self.args.use_mixup):
                mix_alpha = self.args.mixup_alpha
                distribution = torch.distributions.beta.Beta(mix_alpha, mix_alpha)
                lam = distribution.sample()
            
            risk_sum = torch.FloatTensor([0]).cuda()
            mix_loss = torch.FloatTensor([0]).cuda()

            for i in range(self.rels):

                pos1 = logits[(labels[:, i + 1] == 1), i + 1]
                pos2 = logits[(labels[:, i + 1] == 1), 0]
                neg1 = logits[(labels[:, i + 1] != 1), i + 1]
                neg2 = logits[(labels[:, i + 1] != 1), 0]

                if (not eval) and self.args.use_mixup:
                    p_emb_norm = all_emb_norm[(all_labels[:, i + 1] == 1)]
                    p_nums = len(p_emb_norm)
                    p_labels = all_labels[(all_labels[:, i + 1] == 1)]
                    if self.args.mixup_type == 'r_emb':
                        mixemp_norm = lam * p_emb_norm + (1 - lam) * r_emb_norm[0]
                    elif self.args.mixup_type == 'e_emb':
                        n_emb_norm = all_emb_norm[(all_labels[:, i + 1] != 1)]
                        n_nums = len(n_emb_norm)
                        n_idx = random.choices([*range(n_nums)], k=p_nums)
                        n_idx = torch.tensor(n_idx, dtype=torch.long).to(0)
                        n_emb_norm = torch.index_select(n_emb_norm, 0, n_idx)
                        mixemp_norm = lam * p_emb_norm + (1 - lam) * n_emb_norm
                    mix_logits2 = mixemp_norm.matmul(r_emb_norm.T)

                    mix_pos = mix_logits2[(p_labels[:, i + 1] == 1), i + 1]
                    mix_neg = mix_logits2[(p_labels[:, i + 1] == 1), 0]
                    mix_loss += lam * self.loss(mix_pos, mix_neg, self.args.la) + \
                                (1-lam) * self.loss(mix_neg, mix_pos, self.args.la)
                    
                priors_u = (self.priors_o[i] - self.priors_l[i]) / (1. - self.priors_l[i])

                risk1 = (((1. - self.priors_o[i]) / (1. - priors_u)) * self.loss(neg2, neg1, self.args.la) - 
                            ((priors_u - priors_u * self.priors_o[i]) / (1. - priors_u)) * self.loss(pos2, pos1, self.args.la))

                risk2 = self.priors_o[i] * self.loss(pos1, pos2, self.args.la) * self.w[i]

                risk = risk1 + risk2

                if risk1 < self.args.beta:
                    risk = - self.args.gamma * risk1

                risk_sum += risk

            if (not eval) and self.args.use_mixup:
                risk_sum += self.args.mixup_rate * mix_loss

            return risk_sum, logits

        return logits