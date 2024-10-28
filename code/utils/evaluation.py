from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from models.hitercmi import HiterCMI

def train_with_classifiers(data, args):
    model = HiterCMI(args)
    miRNA = data['ms']
    circrna = data['cs']
    train_samples = data['train_samples']
    X = train_samples[:, :2]
    y = train_samples[:, 2]

    mm_matrix = k_matrix(miRNA, args.neighbor)
    cc_matrix = k_matrix(circrna, args.neighbor)


    mm_nx = nx.from_numpy_array(mm_matrix)
    cc_nx = nx.from_numpy_array(cc_matrix)

    mm_graph = dgl.from_networkx(mm_nx)
    cc_graph = dgl.from_networkx(cc_nx)


    mc_copy = copy.deepcopy(data['train_mc'])
    mc_copy[:, 1] = mc_copy[:, 1] + args.miRNA_number
    mc_graph = dgl.graph(
        (np.concatenate((mc_copy[:, 0], mc_copy[:, 1])), np.concatenate((mc_copy[:, 1], mc_copy[:, 0]))),
        num_nodes=args.miRNA_number + args.circrna_number)

    miRNA_th = th.Tensor(miRNA)
    circrna_th = th.Tensor(circrna)


    kf = KFold(n_splits=5, shuffle=True, random_state=args.random_seed)

    fold_index = 0
    results = {}

    for train_index, test_index in kf.split(X):
        print(f"\nFold {fold_index + 1}/5")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


        X_train_samples = train_samples[train_index, :2]
        X_test_samples = train_samples[test_index, :2]

        try:

            embeddings_train = model(mm_graph, cc_graph, mc_graph, miRNA_th, circrna_th, X_train_samples)
            embeddings_test = model(mm_graph, cc_graph, mc_graph, miRNA_th, circrna_th, X_test_samples)

            if embeccings_train is None or embeddings_test is None:
                raise ValueError("模型未返回有效的嵌入特征。请检查模型的 forward 函数。")

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return


        classifiers = {
            "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.6,
                                     colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=-1,
                                     scale_pos_weight=len(y_train) / sum(y_train),
                                     max_delta_step=7)}

        fold_results = {}

        for name, clf in classifiers.items():
            print(f"Training {name} classifier on fold {fold_index + 1}...")
            clf.fit(embeddings_train.detach().numpy(), y_train)
            test_aupr, test_auc = TestOutput(clf, name, embeddings_test.detach().numpy(), y_test, fold_index)
            fold_results[name] = {
                "test_auc": test_auc,
                "test_aupr": test_aupr
            }

        results[f"Fold_{fold_index + 1}"] = fold_results
        fold_index += 1

def TestOutput(classifier, name, X_test, y_test, fold_index):
    ModelTestOutput = classifier.predict_proba(X_test)
    LabelPredictionProb, LabelPrediction = [], []

    for counter in tqdm(range(len(np.array(y_test))), desc=f"Testing {name} output"):
        rowProb = [y_test[counter], ModelTestOutput[counter][1]]
        LabelPrediction.append(row)

    aupr = average_precision_score(y_test, ModelTestOutput[:, 1])
    auc = roc_auc_score(y_test, ModelTestOutput[:, 1])

    return aupr, auc
