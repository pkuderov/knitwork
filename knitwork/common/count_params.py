"""

Counting the number of trainable model parameters.

Use:
  python count_params.py --model rnn   --input_size 27 --output_size 27
  python count_params.py --model grnn  --input_size 27 --output_size 27
  python count_params.py --model grnn_err --input_size 27 --output_size 27
  python count_params.py --model hgrnn --input_size 27 --output_size 27
"""

import argparse
import sys


def build_model(args):
    model_name = args.model
    input_size = args.input_size
    output_size = args.output_size

    embedding_size = args.embedding_size
    base_hidden_size = args.base_hidden_size
    n_layers = args.n_layers
    use_bias = args.use_bias
    dropout = args.dropout

    n_columns = args.n_columns
    n_attn_heads = args.n_attn_heads
    messaging = args.messaging
    col_identities = args.col_identities

    match model_name:
        case 'rnn':
            from knitwork.models.gru import GruBaseline
            model = GruBaseline(
                input_size=input_size,
                embedding_size=embedding_size,
                output_size=output_size,
                base_hidden_size=base_hidden_size,
                hidden_size=args.hidden_size, 
                n_layers=n_layers,
                use_bias=use_bias,
                dropout=dropout,
            )

        case 'grnn':
            from knitwork.models.grnn import GridRnn
            model = GridRnn(
                input_size=input_size,
                embedding_size=embedding_size,
                output_size=output_size,
                base_hidden_size=base_hidden_size,
                n_layers=n_layers,
                n_columns=n_columns,
                n_attn_heads=n_attn_heads,
                messaging=messaging,
                col_identities=col_identities,
                use_bias=use_bias,
                dropout=dropout,
            )

        case 'grnn_err':
            from knitwork.models.grnn_err import GridRnn
            model = GridRnn(
                input_size=input_size,
                embedding_size=embedding_size,
                output_size=output_size,
                base_hidden_size=base_hidden_size,
                n_layers=n_layers,
                n_columns=n_columns,
                n_attn_heads=n_attn_heads,
                messaging=messaging,
                col_identities=col_identities,
                use_bias=use_bias,
                dropout=dropout,
            )

        case 'hgrnn':
            from knitwork.models.hgrnn import HopfieldGridRnn
            model = HopfieldGridRnn(
                input_size=input_size,
                embedding_size=embedding_size,
                output_size=output_size,
                base_hidden_size=base_hidden_size,
                n_layers=n_layers,
                n_columns=n_columns,
                n_attn_heads=n_attn_heads,
                messaging=messaging,
                use_bias=use_bias,
                dropout=dropout,
            )

        case _:
            print(f'Unknown model: {model_name}')
            sys.exit(1)

    return model


def count_parameters(model):
    """return (total_trainable, total_all, per_module_dict)."""
    total_trainable = 0
    total_all = 0
    per_module = {}

    for name, param in model.named_parameters():
        numel = param.numel()
        total_all += numel
        if param.requires_grad:
            total_trainable += numel

        top_module = name.split('.')[0]
        if top_module not in per_module:
            per_module[top_module] = {'trainable': 0, 'total': 0}
        per_module[top_module]['total'] += numel
        if param.requires_grad:
            per_module[top_module]['trainable'] += numel

    return total_trainable, total_all, per_module


def format_num(n):
    if n >= 1_000_000:
        return f'{n:>12,d}  ({n/1e6:.2f}M)'
    elif n >= 1_000:
        return f'{n:>12,d}  ({n/1e3:.1f}K)'
    else:
        return f'{n:>12,d}'


def main():
    parser = argparse.ArgumentParser(
        description='Count trainable parameters of knitwork models'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['rnn', 'grnn', 'grnn_err', 'hgrnn'],
                        help='Model type')
    parser.add_argument('--input_size', type=int, required=True,
                        help='Input vocabulary size (e.g. 27 for text8)')
    parser.add_argument('--output_size', type=int, required=True,
                        help='Output vocabulary size (e.g. 27 for text8)')
    parser.add_argument('--hidden_size', type=int, default=None,
                    help='Direct hidden_size (bypasses convert_hidden_size)')
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--base_hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--use_bias', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--n_columns', type=int, default=2)
    parser.add_argument('--n_attn_heads', type=int, default=4)
    parser.add_argument('--messaging', type=str, default='post',
                        choices=['pre', 'post'])
    parser.add_argument('--col_identities', type=bool, default=True)

    args = parser.parse_args()

    print(f'Model: {args.model}')
    print(f'Input size: {args.input_size}, Output size: {args.output_size}')

    model = build_model(args)

    trainable, total, per_module = count_parameters(model)

    print()
    print('-' * 60)
    print(f'{"Module":<20} {"Trainable":>20} {"Total":>20}')
    print('-' * 60)
    for module_name, counts in per_module.items():
        print(
            f'{module_name:<20}'
            f' {format_num(counts["trainable"]):>20}'
            f' {format_num(counts["total"]):>20}'
        )
    print('-' * 60)
    print(f'{"TOTAL":<20} {format_num(trainable):>20} {format_num(total):>20}')

    # print()
    # print('Detailed parameter list:')
    # print(f'{"Name":<45} {"Shape":<20} {"Numel":>10} {"Grad":>6}')
    # print('-' * 85)
    # for name, param in model.named_parameters():
    #     shape_str = str(list(param.shape))
    #     grad_str = '✓' if param.requires_grad else '✗'
    #     print(f'{name:<45} {shape_str:<20} {param.numel():>10,d} {grad_str:>6}')


if __name__ == '__main__':
    main()