import torch.nn as nn
import torch


def mlp(
    input_dim,
    hidden_dim,
    output_dim,
    hidden_depth,
    output_mod=None,
    batchnorm=False,
    activation=nn.ReLU,
):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = (
            [nn.Linear(input_dim, hidden_dim), activation(inplace=True)]
            if not batchnorm
            else [
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation(inplace=True),
            ]
        )
        for _ in range(hidden_depth - 1):
            mods += (
                [nn.Linear(hidden_dim, hidden_dim), activation(inplace=True)]
                if not batchnorm
                else [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    activation(inplace=True),
                ]
            )
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)

def load_weights_for_mlp(model, path):
    # keyを"_post_grid."があればそれを””に書き換える
    saved_state_dict = torch.load(path)["model"]
    new_state_dict = {}
    for key in saved_state_dict.keys():
        if "_post_grid." in key:
            new_state_dict[key.replace("_post_grid.", "")] = saved_state_dict[key]
        else:
            new_state_dict[key] = saved_state_dict[key]
    model.load_state_dict(new_state_dict)
   
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_depth,
        output_mod=None,
        batchnorm=False,
        activation=nn.ReLU, 
        use_trained_model=False,
        trained_model_path=None,
    ):
        super().__init__()
        self.trunk = mlp(
            input_dim,
            hidden_dim,
            output_dim,
            hidden_depth,
            output_mod,
            batchnorm=batchnorm,
            activation=activation,
        )
        if use_trained_model:
            load_weights_for_mlp(self.trunk, trained_model_path)
            # requires_grad Trueにする
            for param in self.trunk.parameters():
                param.requires_grad = True
        else:
            self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


class ImplicitDataparallel(nn.DataParallel):
    def compute_loss(self, *args, **kwargs):
        return self.module.compute_loss(*args, **kwargs)

    @property
    def temperature(self):
        return self.module.temperature
