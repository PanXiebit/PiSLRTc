import torch
from torch.optim import Optimizer


def build_optimizer(opts, parameters) -> Optimizer:
    """
    Create an optimizer for the given parameters as specified in config.

    Except for the weight decay and initial learning rate,
    default optimizer settings are used.

    Currently supported configuration settings for "optimizer":
        - "sgd" (default): see `torch.optim.SGD`
        - "adam": see `torch.optim.adam`
        - "adagrad": see `torch.optim.adagrad`
        - "adadelta": see `torch.optim.adadelta`
        - "rmsprop": see `torch.optim.RMSprop`

    The initial learning rate is set according to "learning_rate" in the config.
    The weight decay is set according to "weight_decay" in the config.
    If they are not specified, the initial learning rate is set to 3.0e-4, the
    weight decay to 0.

    Note that the scheduler state is saved in the checkpoint, so if you load
    a model for further training you have to use the same type of scheduler.

    :param config: configuration dictionary
    :param parameters:
    :return: optimizer
    """
    optimizer_name = opts.optimizer
    learning_rate = opts.learning_rate   # config.get("learning_rate", 3.0e-4)
    weight_decay = opts.weight_decay     # config.get("weight_decay", 0)
    eps = opts.eps                       # config.get("eps", 1.0e-8)

    # Adam based optimizers
    betas = (0.9, 0.999)                  # config.get("betas", (0.9, 0.999))
    amsgrad = opts.amsgrad               # config.get("amsgrad", False)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adamw":
        return torch.optim.Adam(
            params=parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params=parameters,
            lr=learning_rate,
            lr_decay=opts.lr_decay,   # config.get("lr_decay", 0),
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            params=parameters,
            rho=opts.rho,                        # config.get("rho", 0.9),
            eps=eps,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params=parameters,
            lr=learning_rate,
            momentum=opts.momentum,               # config.get("momentum", 0),
            alpha=opts.alpha,                     # config.get("alpha", 0.99),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params=parameters,
            lr=learning_rate,
            momentum=opts.momentum,               # config.get("momentum", 0),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer {}.".format(optimizer_name))