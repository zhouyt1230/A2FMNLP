import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetname', type=str, default='ckm',
                        help='Dataset name.')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--batch_num', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--epoches', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Whether to use GPU to run (gpu/cpu).')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience.')
    parser.add_argument('--save_checkpoint', type=int, default=1,
                        help='Whether to save model checkpoints (1: save, 0: no).')
    parser.add_argument('--run_test', type=int, default=0,
                        help='Run test only (1: test, 0: train).')
    parser.add_argument('--run_times', type=int, default=10,
                        help='Number of runs.')

    
    parser.add_argument('--inter_aggregation', type=str, default='logit',
                        help='Legacy arg, kept for compatibility.')

    
    parser.add_argument('--run_bucket_eval', type=int, default=1,
                        help='Whether to run degree-bucket evaluation on the test set.')
    parser.add_argument('--bucket_mode', type=str, default='min',
                        choices=['min', 'max', 'mean'],
                        help='How to convert two endpoint degrees into one edge degree.')
    parser.add_argument('--bucket_batch_num', type=int, default=1024,
                        help='Batch size for bucket evaluation.')

    
    parser.add_argument('--model_ablation', type=str, default='full',
                        choices=['full', 'wogtn', 'womlp'],
                        help='Model ablation setting.')

   
    parser.add_argument('--fusion_type', type=str, default='gate',
                        choices=['add', 'gate'],
                        help='Inter-layer fusion type.')

    return parser.parse_args()
