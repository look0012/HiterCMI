import argparse
import os
from config import get_args
from utils.data_processing import get_data, data_processing
from utils.evaluation import train_with_classifiers


def main():
    args = get_args()

    # 确保结果目录存在
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    data = get_data(args)
    args.miRNA_number = data['miRNA_number']
    args.circrna_number = data['circrna_number']
    data_processing(data, args)

    # 调用训练方法
    results = train_with_classifiers(data, args)

    # 保存最终结果
    result_dir = args.result_dir
    with open(os.path.join(result_dir, 'classifier_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("Training complete. Results saved.")


if __name__ == "__main__":
    main()
