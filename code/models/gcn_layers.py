import torch as th
from torch import nn
from dgl import function as fn

class ConvLayer(nn.Module):
    def __init__(self, in_feats, out_feats, k=2, method='sum', bias=True, batchnorm=False, activation='relu', dropout=0.0):
        super(ConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k + 1
        self.method = method
        self.weights = []
        for i in range(self.k):
            self.weights.append(nn.Parameter(th.Tensor(in_feats, out_feats)))
        self.biases = None
        self.activation = None
        self.batchnorm = None
        self.dropout = None
        if bias:
            self.biases = []
            for i in range(self.k):
                self.biases.append(nn.Parameter(th.Tensor(out_feats)))
        self.reset_parameters()

        if activation == 'relu':
            self.activation = th.relu
        if batchnorm:
            if method == 'cat':
                self.batchnorm = nn.BatchNorm1d(out_feats * self.k)
            else:
                self.batchnorm = nn.BatchNorm1d(out_feats)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for i in range(self.k):
            nn.init.xavier_uniform_(self.weights[i])
            if self.biases is not None:
                nn.init.zeros_(self.biases[i])

    def forward(self, graph, feat):
        with graph.local_scope():
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)

            if self.biases is not None:
                rst = th.matmul(feat, self.weights[0]) + self.biases[0]
            else:
                rst = th.matmul(feat, self.weights[0])

            for i in range(1, self.k):
                feat = feat * norm
                graph.ndata['h'] = feat
                if 'e' in graph.edata.keys():
                    graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

                if self.method == 'sum':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = rst + y
                elif self.method == 'mean':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = rst + y
                    rst = rst / self.k
                elif self.method == 'cat':
                    if self.biases is not None:
                        y = th.matmul(feat, self.weights[0]) + self.biases[0]
                    else:
                        y = th.matmul(feat, self.weights[0])
                    rst = th.cat((rst, y), dim=1)

            if self.batchnorm is not None:
                rst = self.batchnorm(rst)
            if self.activation is not None:
                rst = self.activation(rst)
            if self.dropout is not None:
                rst = self.dropout(rst)
            return rst
