import argparse
import collections
import torch
import os
import re


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state
        # print(state.keys())
        # exit()
        model_params = state['model_state_dict']

        model_params_keys = list(model_params.keys())

        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    new_state['model_state_dict'] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r'ep\d+_wer_(\d+.?\d+)_bleu_(\d+.?\d+).pkl')
    else:
        pt_regexp = re.compile(r'ep\d+_wer_(\d+.?\d+)_bleu_(\d+.?\d+).pkl')
    files = os.listdir(path)
    # print("files: ", files)
    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        print("file: ", f)
        print("pattern: ", m)
        if m is not None:
            sort_key = int(m.group(0)[2:4])
            print("sort_key: ", sort_key)
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))

    if len(entries) < n:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)

    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-epoch-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    num_group.add_argument('--num-update-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which checkpoint to use, '
                        'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be averaged.')
    # fmt: on
    args = parser.parse_args()
    print(args)

    num = None
    is_update_based = False
    if args.num_update_checkpoints is not None:
        num = args.num_update_checkpoints
        is_update_based = True
    elif args.num_epoch_checkpoints is not None:
        num = args.num_epoch_checkpoints

    assert args.checkpoint_upper_bound is None or args.num_epoch_checkpoints is not None, \
            '--checkpoint-upper-bound requires --num-epoch-checkpoints'
    assert args.num_epoch_checkpoints is None or args.num_update_checkpoints is None, \
            'Cannot combine --num-epoch-checkpoints and --num-update-checkpoints'

    if num is not None:
        args.inputs = last_n_checkpoints(
            args.inputs, num, is_update_based, upper_bound=args.checkpoint_upper_bound,
        )
        print('averaging checkpoints: ', args.inputs)

    new_state = average_checkpoints(args.inputs)
    torch.save(new_state, args.output)
    print('Finished writing averaged checkpoint to {}.'.format(args.output))


if __name__ == '__main__':
    main()