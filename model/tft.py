import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union

import numpy as np

from torchmetrics import Metric as LightningMetric

from pytorch_forecasting import create_mask
from pytorch_forecasting.models.nn import LSTM
from pytorch_forecasting.models import BaseModelWithCovariates
from pytorch_forecasting.metrics import (
    MAPE,
    MAE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
    QuantileLoss,
)
from model.attention import InterpretableMultiHeadAttention
from model.embeddings import MultiEmbedding
from model.layers import (
    GatedLinearUnit,
    AddNorm,
    GatedAddNorm,
    GatedResidualNetwork,
    VariableSelectionNetwork,
)


class TemporalFusionTransformer(BaseModelWithCovariates):
    def __init__(
        self,
        hidden_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        output_size: Union[int, List[int]] = 7,
        loss: MultiHorizonMetric = None,  # must pass this up to BaseModelWithCovariates -> BaseModel to override default
        attention_head_size: int = 4,
        max_encoder_length: int = 10,
        static_categoricals: List[str] = [],
        static_reals: List[str] = [],
        time_varying_categoricals_encoder: List[str] = [],
        time_varying_categoricals_decoder: List[str] = [],
        categorical_groups: Dict[str, List[str]] = {},
        time_varying_reals_encoder: List[str] = [],
        time_varying_reals_decoder: List[str] = [],
        x_reals: List[str] = [],
        x_categoricals: List[str] = [],
        hidden_continuous_size: int = 8,
        hidden_continuous_sizes: Dict[str, int] = {},
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        embedding_paddings: List[str] = [],
        embedding_labels: Dict[str, np.ndarray] = {},
        learning_rate: float = 1e-3,
        log_interval: Union[int, float] = -1,
        log_val_interval: Union[int, float] = None,
        log_gradient_flow: bool = False,
        reduce_on_plateau_patience: int = 1000,
        monotone_constaints: Dict[str, int] = {},
        share_single_variable_networks: bool = False,
        causal_attention: bool = True,
        logging_metrics: nn.ModuleList = None,  # same here as loss
        **kwargs,
    ):

        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            loss = QuantileLoss()

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])

        assert isinstance(
            loss, LightningMetric
        ), "Loss must be Pytorch Lightning 'Metric'"
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        # start to process inputs gl

        # this gets the embedding tensors, not sure what the shape is yet
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
            max_embedding_size=self.hparams.hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(
                    1,
                    self.hparams.hidden_continuous_sizes.get(
                        name, self.hparams.hidden_continuous_size
                    ),
                )
                for name in self.reals  # from base model -> """List of all continuous variables in model"""
            }  # static_categoricals + time_varying_categorical_encoder + time_varying_categorical_decoder,
        )  # {name:Linear layer(1,10)} for all continuous vars. hidden_continuous_sizes can be a dict of different sizes

        # variable selection

        ## variable selection for static variables
        static_input_sizes = (
            {  # not defined as an attribute just a local variable to be used shortly
                name: self.input_embeddings.output_size[name]
                for name in self.hparams.static_categoricals
            }
        )  # {'fueltype':5} for my example

        static_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.static_reals
            }
        )  # {'encoder_length':10, 'value_center':10,'value_scale':10}

        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.static_categoricals
            },
            dropout=self.hparams.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        }

        encoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: self.hparams.hidden_continuous_sizes.get(
                    name, self.hparams.hidden_continuous_size
                )
                for name in self.hparams.time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if self.hparams.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hparams.hidden_size),
                    self.hparams.hidden_size,
                    self.hparams.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hparams.hidden_size),
                        self.hparams.hidden_size,
                        self.hparams.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_encoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )
        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hparams.hidden_size,
            input_embedding_flags={
                name: True for name in self.hparams.time_varying_categoricals_decoder
            },
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns=(
                {}
                if not self.hparams.share_single_variable_networks
                else self.shared_single_variable_grns
            ),
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )
        # for hidden state of lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )
        # for cell state of lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )
        # for post lstm static enrichment

        self.static_context_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # lstm encoder(history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            num_layers=self.hparams.lstm_layers,
            dropout=self.hparams.dropout if self.hparams.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm

        self.post_lstm_gate_encoder = GatedLinearUnit(
            self.hparams.hidden_size, dropout=self.hparams.dropout
        )
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder

        self.post_lstm_add_norm_encoder = AddNorm(
            self.hparams.hidden_size, trainable_add=False
        )

        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past lstm
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size,
        )
        # attention for long range processing

        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hparams.hidden_size,
            n_head=self.hparams.attention_head_size,
            dropout=self.hparams.dropout,
        )
        self.post_attn_gate_norm = GatedAddNorm(
            self.hparams.hidden_size, dropout=self.hparams.dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            self.hparams.hidden_size,
            dropout=self.hparams.dropout,
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GatedAddNorm(
            self.hparams.hidden_size, dropout=None, trainable_add=False
        )

        if self.n_targets > 1:
            self.output_layer = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size, output_size)
                    for output_size in self.hparams.output_size
                ]
            )
        else:
            self.output_layer = nn.Linear(
                self.hparams.hidden_size, self.hparams.output_size
            )

    def expand_static_context(self, context, timesteps):
        """
        add dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(
        self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor
    ):
        """
        returns causal mask to apply for self-attention layer
        """

        decoder_length = decoder_lengths.max()

        if self.hparams.causal_attention:
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=self.device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = (
                (attend_step >= predict_step)
                .unsqueeze(0)
                .expand(encoder_lengths.size(0), -1, -1)
            )
        else:
            """
            there is value in attending to future forecasts if they are made with knowledge currently available
            one possibility is here to use a second attention layer for future attention (assuming different
            effects matter in the future than the past)
            or alternatively using the same layer but allowing forward attention - only masking out non-available data
            and self
            """
            decoder_mask = (
                create_mask(decoder_length, decoder_lengths)
                .unsqueeze(1)
                .expand(-1, decoder_length, -1)
            )
        # do not attend to steps where data is padded
        encoder_mask = (
            create_mask(encoder_lengths.max(), encoder_lengths)
            .unsqueeze(1)
            .expand(-1, decoder_length, -1)
        )

        # combine masks along attended time - encoder then decoder

        mask = torch.cat((encoder_mask, decoder_mask), dim=2)

        return mask


def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    encoder_lengths = x["encoder_lengths"]
    decoder_lengths = x["decoder_lengths"]
    x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
    x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
    timesteps = x_cont.size(1)
    max_encoder_length = int(encoder_lengths.max())
    input_vectors = self.input_embeddings(x_cat)
    input_vectors.update(
        {
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.hparams.x_reals)
            if name in self.reals
        }
    )
    static_embedding, static_variable_selection = (
        (
            torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            ),
            torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device),
        )
        if len(self.static_variables) == 0
        else self.static_variable_selection(
            {name: input_vectors[name][:, 0] for name in self.static_variables}
        )
    )
    static_context_variable_selection = self.expand_static_context(
        self.static_context_variable_selection(static_embedding), timesteps
    )
    embeddings_varying_encoder, encoder_sparse_weights = (
        self.encoder_variable_selection(
            {
                name: input_vectors[name][:, :max_encoder_length]
                for name in self.encoder_variables
            },
            static_context_variable_selection[:, :max_encoder_length],
        )
    )
    embeddings_varying_decoder, decoder_sparse_weights = (
        self.decoder_variable_selection(
            {
                name: input_vectors[name][:, max_encoder_length:]
                for name in self.decoder_variables
            },
            static_context_variable_selection[:, max_encoder_length:],
        )
    )
    input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
        self.hparams.lstm_layers, -1, -1
    )
    input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
        self.hparams.lstm_layers, -1, -1
    )
    encoder_output, (hidden, cell) = self.lstm_encoder(
        embeddings_varying_encoder,
        (input_hidden, input_cell),
        lengths=encoder_lengths,
        enforce_sorted=False,
    )
    decoder_output, _ = self.lstm_decoder(
        embeddings_varying_decoder,
        (hidden, cell),
        lengths=decoder_lengths,
        enforce_sorted=False,
    )
    lstm_output_encoder = self.post_lstm_add_norm_encoder(
        self.post_lstm_gate_encoder(encoder_output), embeddings_varying_encoder
    )
    lstm_output_decoder = self.post_lstm_add_norm_decoder(
        self.post_lstm_gate_encoder(decoder_output), embeddings_varying_decoder
    )
    lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
    static_context_enrichment = self.static_context_enrichment(static_embedding)
    attn_input = self.static_enrichment(
        lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
    )
    attn_output, attn_output_weights = self.multihead_attn(
        q=attn_input[:, max_encoder_length:],
        k=attn_input,
        v=attn_input,
        mask=self.get_attention_mask(
            encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
        ),
    )
    attn_output = self.post_attn_gate_norm(
        attn_output, attn_input[:, max_encoder_length:]
    )
    output = self.pos_wise_ff(attn_output)
    output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
    output = (
        [output_layer(output) for output_layer in self.output_layer]
        if self.n_targets > 1
        else self.output_layer(output)
    )
    return self.to_network_output(
        prediction=self.transform_output(output, target_scale=x["target_scale"]),
        encoder_attention=attn_output_weights[..., :max_encoder_length],
        decoder_attention=attn_output_weights[..., max_encoder_length:],
        static_variables=static_variable_selection,
        encoder_variables=encoder_sparse_weights,
        encoder_lengths=encoder_lengths,
        decoder_lengths=decoder_lengths,
    )
