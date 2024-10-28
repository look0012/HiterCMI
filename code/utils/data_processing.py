import numpy as np
import random
from utils.graph_operations import make_adj, get_edge_index

def get_data(args):
    data = dict()
    mf = np.loadtxt(args.data_dir + 'mi_fused_matrix.txt', dtype=np.float64)
    mfw = np.loadtxt(args.data_dir + 'mi_fused_matrix_weight.txt', dtype=np.float64)
    css = np.loadtxt(args.data_dir + 'circ_fused_matrix.txt', dtype=np.float64)
    csw = np.loadtxt(args.data_dir + 'circ_fused_matrix_weight.txt', dtype=np.float64)

    data['miRNA_number'] = int(mf.shape[0])
    data['circrna_number'] = int(dss.shape[0])
    data['mf'] = mf
    data['css'] = css
    data['mfw'] = mfw
    data['csw'] = csw
    data['c_num'] = np.loadtxt('circ_number.txt', delimiter='\t', dtype=np.str_)[:, 1]
    data['m_num'] = np.loadtxt('mi_number.txt', delimiter='\t', dtype=np.str_)[:, 1]
    data['mc'] = np.loadtxt('9905pairs_number.txt', dtype=np.int32) - 1
    return data

def data_processing(data, args):
    mc_matrix = make_adj(data['mc'], (args.miRNA_number, args.circrna_number))
    one_index = []
    zero_index = []

    for i in range(mc_matrix.shape[0]):
        for j in range(mc_matrix.shape[1]):
            if mc_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])

    random.seed(args.random_seed)
    random.shuffle(one_index)
    unsamples = []
    if args.negative_rate == -1:
        zero_index = zero_index
    else:
        unsamples = zero_index[int(args.negative_rate * len(one_index)):]
        zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)

    mc = samples[samples[:, 2] == 1, :2]
    mc_matrix = make_adj(mc, (args.miRNA_number, args.circrna_number))
    mc_matrix = mc_matrix.numpy()

    ms = data['mf'] * data['mfw']
    ds = data['dss'] * data['dsw']

    data['ms'] = ms
    data['cs'] = ds
    data['train_samples'] = samples
    data['train_mc'] = mc
    data['unsamples'] = np.array(unsamples)
