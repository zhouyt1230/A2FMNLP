import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloder import construct_loader, get_weak_info, get_degree_bucket_info, get_dataset
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from model.MWTPModel import A2FMNLPModel
from args import parse_args
import time
from itertools import cycle, islice
from itertools import zip_longest


def print_bucket_markdown_table(bucket_results):
    if not bucket_results:
        return

    print("\nBucket evaluation markdown table:")
    print("| Bucket | Samples | Acc | AP | F1 | AUC |")
    print("|---|---:|---:|---:|---:|---:|")
    for bucket_name, metrics in bucket_results.items():
        print(
            f"| {bucket_name} | {metrics['samples']} | "
            f"{metrics['acc']:.4f} | {metrics['ap']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['auc']:.4f} |"
        )
    print("")


def train(ori_adj, break_adj, feats, idx_train, idx_val, idx_test,
          layer_num, num_nodes, args, device, times):

    labels = []
    ori_labels = []
    for l in range(layer_num):
        labels.append(torch.Tensor(break_adj[l].flatten()))
        ori_labels.append(torch.Tensor(ori_adj[l].flatten()))

    adj_lists = []
    for ki in range(layer_num):
        adj_lists_temp = defaultdict(set)
        [row, col] = np.where(break_adj[ki] == 1)
        for i in range(row.size):
            adj_lists_temp[row[i]].add(col[i])
            adj_lists_temp[col[i]].add(row[i])
        adj_lists.append(adj_lists_temp)

    weak_eval_data = get_weak_info(
        args.datasetname, idx_train, idx_val, idx_test,
        ori_adj, ori_labels, layer_num, num_nodes
    )

    bucket_eval_data = None
    if args.run_bucket_eval:
        bucket_eval_data = get_degree_bucket_info(
            idx_test=idx_test,
            ori_adj=ori_adj,
            labels=ori_labels,
            layer_num=layer_num,
            num_nodes=num_nodes,
            batch_num=args.bucket_batch_num,
            bucket_mode=args.bucket_mode
        )

    model = MWTPModel(
        layer_num, num_nodes, feats, break_adj, adj_lists, args.emb_dim, device,
        fusion_type=args.fusion_type,
        ablation=args.model_ablation
    )

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    train_data_loaders, max_len_train = construct_loader(idx_train, labels, layer_num, num_nodes, args.batch_num)
    val_data_loaders, max_len_val = construct_loader(idx_val, labels, layer_num, num_nodes, args.batch_num)
    test_data_loaders, max_len_test = construct_loader(idx_test, ori_labels, layer_num, num_nodes, args.batch_num)

    if args.run_test:
        print('Start Test Step')
        model.load_state_dict(torch.load(f'checkpoint/{args.datasetname}_{times}_model_checkpoint.pth'))
        test(model, test_data_loaders, max_len_test, weak_eval_data,
             bucket_eval_data=bucket_eval_data, eval_type='test')
        return

    print('Start Train Step')

    auc_patience = 0
    best_auc = -float('inf')

    for epoch in range(args.epoches):
        model.train()
        epoches_times, epoches_loss, epoches_acc, epoches_auc, epoches_ap, epoches_f1 = [], [], [], [], [], []

        for batch_idx, *batch_data in tqdm(
                enumerate(zip(*[islice(cycle(loader), max_len_train) for loader in train_data_loaders])),
                desc='processing', unit='items', total=max_len_train):

            loss, acc, auc, ap, f1 = [], [], [], [], []
            start_train_time = time.time()

            for train_layer, train_data in enumerate(*batch_data):
                train_nodes, train_targets = train_data
                loss_t, acc_t, auc_t, ap_t, f1_t = model(train_nodes.view(-1), train_targets, train_layer)
                loss.append(loss_t)
                acc.append(acc_t)
                auc.append(auc_t)
                ap.append(ap_t)
                f1.append(f1_t)

            if len(loss) != 0:
                loss = sum(loss) / len(loss)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

                epoches_times.append(time.time() - start_train_time)
                epoches_loss.append(loss)
                epoches_acc.append(sum(acc) / len(acc))
                epoches_auc.append(sum(auc) / len(auc))
                epoches_ap.append(sum(ap) / len(ap))
                epoches_f1.append(sum(f1) / len(f1))

        print("train epoches: {}/{} | ".format(epoch + 1, args.epoches),
              "loss: {:.4f} | ".format(sum(epoches_loss) / len(epoches_loss)),
              "acc: {:.4f} | ".format(sum(epoches_acc) / len(epoches_acc)),
              "ap: {:.4f} | ".format(sum(epoches_ap) / len(epoches_ap)),
              "f1: {:.4f} | ".format(sum(epoches_f1) / len(epoches_f1)),
              "auc: {:.4f} | ".format(sum(epoches_auc) / len(epoches_auc)),
              "time: {:.4f}".format(sum(epoches_times)))

        eval(model, val_data_loaders, max_len_val, eval_type='eval')

        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                patience_auc = test(
                    model, test_data_loaders, max_len_test, weak_eval_data,
                    bucket_eval_data=bucket_eval_data, eval_type='test'
                )

            if patience_auc > best_auc:
                auc_patience = 0
                best_auc = patience_auc
            else:
                auc_patience += 1
                if auc_patience > args.patience:
                    print("Early Stopping")
                    break

        torch.cuda.empty_cache()

        if args.save_checkpoint:
            os.makedirs('checkpoint', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoint/{args.datasetname}_{times}_model_checkpoint.pth')
            print("Model parameters saved!")


def test(model, test_data_loaders, max_len_loader, weak_eval_data,
         bucket_eval_data=None, eval_type='test'):
    print('----------------------------Start Eval Step--------------------------------------')

    overall_metrics = eval(model, test_data_loaders, max_len_loader,
                           eval_type='Overall Ties', return_metrics=True)

    weak_loaders, weak_info = weak_eval_data
    eval(model, weak_loaders[0], weak_loaders[1], data_info=weak_info,
         eval_type='Weak Ties', return_metrics=False)

    if bucket_eval_data is not None:
        bucket_results = {}
        for bucket_name, bucket_pack in bucket_eval_data.items():
            metrics = eval(
                model,
                bucket_pack['data'][0],
                bucket_pack['data'][1],
                data_info=bucket_pack['info'],
                eval_type=bucket_name,
                return_metrics=True
            )
            if metrics is None:
                continue
            metrics['samples'] = int(sum(sum(x) for x in bucket_pack['info']))
            bucket_results[bucket_name] = metrics

        print_bucket_markdown_table(bucket_results)

    print('----------------------------End Eval Step--------------------------------------')

    if eval_type == 'test':
        return overall_metrics['auc']


def eval(model, data_loaders, max_len_loader, data_info=None, eval_type='', return_metrics=False):
    model.eval()
    with torch.no_grad():
        batches_times, batches_loss, batches_acc, batches_auc, batches_ap, batches_f1 = [], [], [], [], [], []

        for batch_idx, *batch_data in enumerate(zip_longest(*data_loaders, fillvalue=None)):
            loss, acc, auc, ap, f1 = [], [], [], [], []
            start_train_time = time.time()

            for eval_layer, eval_data in enumerate(*batch_data):
                if eval_data is None:
                    continue
                if data_info and (data_info[eval_layer][0] == 0 or data_info[eval_layer][1] == 0):
                    continue

                eval_nodes, eval_targets = eval_data
                loss_t, acc_t, auc_t, ap_t, f1_t = model(eval_nodes.view(-1), eval_targets, eval_layer)
                loss.append(loss_t)
                acc.append(acc_t)
                auc.append(auc_t)
                ap.append(ap_t)
                f1.append(f1_t)

            if len(loss) != 0:
                batches_times.append(time.time() - start_train_time)
                batches_loss.append(sum(loss) / len(loss))
                batches_acc.append(sum(acc) / len(acc))
                batches_auc.append(sum(auc) / len(auc))
                batches_ap.append(sum(ap) / len(ap))
                batches_f1.append(sum(f1) / len(f1))

        if len(batches_loss) == 0:
            print(f"{eval_type}: skipped (no valid positive/negative pairs in this bucket).")
            return None

        metrics = {
            'loss': float(sum(batches_loss) / len(batches_loss)),
            'acc': float(sum(batches_acc) / len(batches_acc)),
            'ap': float(sum(batches_ap) / len(batches_ap)),
            'f1': float(sum(batches_f1) / len(batches_f1)),
            'auc': float(sum(batches_auc) / len(batches_auc)),
            'time': float(sum(batches_times))
        }

        print("{} loss: {:.4f} | ".format(eval_type, metrics['loss']),
              "acc: {:.4f} | ".format(metrics['acc']),
              "ap: {:.4f} | ".format(metrics['ap']),
              "f1: {:.4f} | ".format(metrics['f1']),
              "auc: {:.4f} | ".format(metrics['auc']),
              "time: {:.4f}".format(metrics['time']))

        if return_metrics:
            return metrics

        if eval_type == 'Overall Ties':
            return round(metrics['auc'], 4)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.device == 'gpu':
        cuda = torch.cuda.is_available()
        device = torch.device("cuda:0") if cuda else torch.device("cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    else:
        raise Exception("Invalid device option. Please choose 'gpu' or 'cpu'.")

    ori_adj, break_adj, feats, idx_train, idx_val, idx_test = get_dataset(args.datasetname)
    layer_num = ori_adj.shape[0]
    num_nodes = ori_adj.shape[1]

    print('adj shape: ', ori_adj.shape)
    print('layer num: ', layer_num)
    print('nodes num: ', num_nodes)

    for times in range(args.run_times):
        print(f'The times: {times}')
        train(ori_adj, break_adj, feats, idx_train, idx_val, idx_test,
              layer_num, num_nodes, args, device, times)
        print('----------------------------------------------------------------------------------')
