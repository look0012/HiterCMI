import torch as th
from torch import nn
from models.gcn_layers import ConvLayer

class GraphEmbedding(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, k, method, bias, batchnorm, activation, num_layers, dropout):
        super(GraphEmbedding, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                hid_feats = out_feats
            self.layers.append(ConvLayer(in_feats, hid_feats, k, method, bias, batchnorm, activation, dropout))
            if method == 'cat':
                in_feats = hid_feats * (k + 1)
            else:
                in_feats = hid_feats

    def forward(self, graph, feat):
        for i, layer in enumerate(self.layers):
            feat = layer(graph, feat)
        return feat

class HiterCMI(nn.Module):
    def __init__(self, args):
        super(HiterCMI, self).__init__()
        self.args = args
        self.lin_m = nn.Linear(args.miRNA_number, args.in_feats, bias=False)
        self.lin_d = nn.Linear(args.circrnacircrna_number, args.in_feats, bias=False)

        self.gcn_mm = GraphEmbedding(args.miRNA_number, args.hid_feats, args.out_feats, args.k, args.method,
                                     args.gcn_bias, args.gcn_batchnorm, args.gcn_activation, args.num_layers,
                                     args.dropout)
        self.gcn_cc = GraphEmbedding(args.circrna_number, args.hid_feats, args.out_feats, args.k, args.method,
                                     args.gcn_bias, args.gcn_batchnorm, args.gcn_activation, args.num_layers,
                                     args.dropout)
        self.gcn_mc = GraphEmbedding(args.in_feats, args.hid_feats, args.out_feats, args.k, args.method,
                                     args.gcn_bias, args.gcn_batchnorm, args.gcn_activation, args.num_layers,
                                     args.dropout)

    def forward(self, mm_graph, cc_graph, mc_graph, miRNA, circrna, samples):
        emb_mm_sim = self.gcn_mm(mm_graph, miRNA)
        emb_cc_sim = self.gcn_cc(cc_graph, circrna)

        emb_ass = self.gcn_mc(mc_graph, th.cat((self.lin_m(miRNA), self.lin_d(circrna)), dim=0))
        emb_mm_ass = emb_ass[:self.args.miRNA_number, :]
        emb_cc_ass = emb_ass[self.args.miRNA_number:, :]
        emb_mm = th.cat((emb_mm_sim, emb_mm_ass), dim=1)
        emb_cc = th.cat((emb_cc_sim, emb_cc_ass), dim=1)
        emb = th.cat((emb_mm[samples[:, 0]], emb_cc[samples[:, 1]]), dim=1)

        return emb
