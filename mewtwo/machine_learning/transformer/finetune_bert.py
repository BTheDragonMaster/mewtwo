import argparse

from transformers import AutoTokenizer, AutoModel

from mewtwo.external_code.dnabert.bert_layers import BertModel
from mewtwo.machine_learning.transformer.dnabert_regressor import DNABERTRegressor
from torch.utils.data import DataLoader
from mewtwo.machine_learning.transformer.prepare_data import RegressionDataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, help="Tabular input data, with sequence in one column and efficiency in the second")
    parser.add_argument("-o", type=str, help="Output directoru")
    parser.add_argument("-f", type=str, default='linear_head',
                        help="Finetuning mode, must be one of 'linear_head', 'partial', and 'adapter'")
    args = parser.parse_args()

    assert args.f in ['linear_head', 'partial', 'adapter']

    return args

if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")
    model.to("cpu")

    dataset = RegressionDataset(sequences, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for name, param in model.named_parameters():
        print(name, param)
