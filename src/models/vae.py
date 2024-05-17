from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiVAE(nn.Module):
    def __init__(
        self,
        item_num: int,
        latent_dim: int = 200,
        num_hidden: int = 1,
        hidden_dim: int = 600,
        dropout: float = 0.5,
    ) -> None:
        """MultVAE implementation (Variational Autoencoders for Collaborative Filtering)
            https://arxiv.org/abs/1802.05814

        Args:
            item_num (int): Number of items.
            latent_dim (int, optional): Dimension of the latent representation. Defaults to 200.
            num_hidden (int, optional): Number of hidden layers in each encoder and decoder MLP layers. Defaults to 1.
            hidden_dim (int, optional): Dimension of hidden layers in encoder and decoder MLP layers. Defaults to 600.
            dropout (float, optional): Defaults to 0.1.

        """

        super().__init__()

        self.latent_dim = latent_dim
        self.item_num = item_num
        hidden_dims = [hidden_dim] * num_hidden

        encoder_dims = [self.item_num] + hidden_dims + [self.latent_dim * 2]
        decoder_dims = [self.latent_dim] + hidden_dims + [self.item_num]

        self.encoder = nn.ModuleList(
            [
                nn.Linear(dim_in, dim_out)
                for dim_in, dim_out in zip(encoder_dims[:-1], encoder_dims[1:])
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Linear(dim_in, dim_out)
                for dim_in, dim_out in zip(decoder_dims[:-1], decoder_dims[1:])
            ]
        )

        self.dropout = nn.Dropout(p=dropout)

        for layer in self.encoder:
            self._init_weight(layer)

        for layer in self.decoder:
            self._init_weight(layer)

    def _init_weight(self, layer : nn.Linear) -> None:
        """
        Xavier initialization

        Args:
            layer: layer of a model
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data)
            layer.bias.data.normal_(0.0, 0.001)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.normalize(x, p=2, dim=1)
        print(hidden)
        hidden = self.dropout(hidden)
        print(hidden)

        for layer in self.encoder[:-1]:
            hidden = layer(hidden)
            print(hidden)
            hidden = torch.tanh(hidden)

        hidden = self.encoder[-1](hidden)

        mu = hidden[:, : self.latent_dim]
        logvar = hidden[:, self.latent_dim :]

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick"""

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = z

        for layer in self.decoder[:-1]:
            hidden = layer(hidden)
            hidden = torch.tanh(hidden)

        return self.decoder[-1](hidden)

    def forward(self, batch : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(batch)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
