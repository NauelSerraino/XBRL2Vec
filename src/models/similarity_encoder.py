"""
Encoder for the similarity benchmark: the DLinear + FiLM encoder of ForecastingAE, without its decoder, and
optionally a small MLP between the conditioned features and the embedding (the linear encoder cannot combine
variables non-linearly, e.g. form a ratio of two lines).

hidden = 0 and film_init = "default" reproduce ForecastingAE's encoder exactly (fin DLinear -> FiLM by the macro
DLinear -> linear to z). film_init = "identity" uses IdentityFiLM instead: gamma = 1 + dgamma, beta = dbeta with the
layer producing (dgamma, dbeta) initialised at zero, so the untrained model ignores the macro and the macro only
enters if it lowers the loss (with the default FiLM, an MLP model could let the macro take over the representation).
"""
import torch
import torch.nn as nn

from models.autoencoder_dlinear_forecaster import DLinearEncoder, MacroConditioner


class IdentityFiLM(nn.Module):
    """FiLM conditioner that starts as the identity: returns (1 + dgamma, dbeta), both 0 at initialisation."""

    def __init__(self, M, fin_dim):
        super().__init__()
        self.net = nn.Linear(M, 2 * fin_dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, h_macro):
        dgamma, beta = self.net(h_macro).chunk(2, dim=-1)
        return 1.0 + dgamma, beta


class MacroCrossAttention(nn.Module):
    """
    Firm-specific read of the macro window (alternative to FiLM): the firm's encoding is the query, the macro quarters
    (with a learned position embedding) the keys and values; the result is added to the firm's encoding through a
    zero-initialised projection, so at initialisation the macro has no effect (as IdentityFiLM).
    """

    def __init__(self, fin_dim: int, M: int, T: int, d: int = 32, uniform: bool = False):
        super().__init__()
        self.uniform = uniform                    # True: every quarter weighted 1/T (an additive mean-pooled macro read)
        self.q, self.k, self.v = nn.Linear(fin_dim, d), nn.Linear(M, d), nn.Linear(M, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        self.out = nn.Linear(d, fin_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.scale = d ** -0.5

    def forward(self, h, x_macro):
        k = self.k(x_macro) + self.pos                                 # [B, T, d]
        a = torch.softmax((self.q(h)[:, None, :] * k).sum(-1) * self.scale, dim=1)  # [B, T]: which quarters matter
        if self.uniform:
            a = torch.full_like(a, 1.0 / a.shape[1])
        return h + self.out((a[..., None] * self.v(x_macro)).sum(1))


class SimilarityEncoder(nn.Module):
    def __init__(self, T: int, F: int, M: int, latent_dim: int, use_macro: bool = True, hidden: int = 0,
                 film_init: str = "default", macro_dropout: float = 0.0, conditioning: str = "film",
                 macro_target: str = "both"):
        super().__init__()
        self.use_macro = use_macro
        self.conditioning = conditioning
        self.macro_target = macro_target          # "both": z uses the macro; "pillar2": z is macro-free and the
                                                  # macro-conditioned embedding is returned second (Pillar 2 only)
        self.fin_encoder = DLinearEncoder(T, F)
        if use_macro and conditioning in ("cross_attention", "mean_pool"):
            self.cross = MacroCrossAttention(2 * F, M, T, uniform=(conditioning == "mean_pool"))
            self.macro_dropout = nn.Dropout(macro_dropout)
        elif use_macro:
            self.macro_encoder = DLinearEncoder(T, M)
            self.macro_dropout = nn.Dropout(macro_dropout)             # on the macro embedding (training only)
            self.conditioner = {"default": MacroConditioner, "identity": IdentityFiLM}[film_init](2 * M, 2 * F)
        if hidden:
            self.to_latent = nn.Sequential(nn.Linear(2 * F, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                           nn.Linear(hidden, latent_dim))
        else:
            self.to_latent = nn.Linear(2 * F, latent_dim)

    def forward(self, x_fin, x_macro=None):
        h = self.fin_encoder(x_fin)                                  # [B, 2F]
        hm = h
        if self.use_macro and x_macro is not None and self.conditioning in ("cross_attention", "mean_pool"):
            hm = self.cross(h, self.macro_dropout(x_macro))
        elif self.use_macro and x_macro is not None:
            gamma, beta = self.conditioner(self.macro_dropout(self.macro_encoder(x_macro)))
            hm = gamma * h + beta
        if self.macro_target == "pillar2" and self.use_macro and x_macro is not None:
            return self.to_latent(h), self.to_latent(hm)             # (macro-free z, macro-conditioned z for Pillar 2)
        return self.to_latent(hm), None                              # (z, no reconstruction)


class ScenarioHead(nn.Module):
    """
    Scenario-conditioned Pillar 2 (docs/pillar1_benchmark.md): reweights the latent dimensions by the realised macro
    scenario s of the target period, z̃ = (1 + W s + b) ⊙ z. W and b start at zero, so z̃ = z until training finds a
    reweighting that helps. Used only in the Pillar 2 term of the loss: z itself never sees the scenario.
    """

    def __init__(self, S: int, latent_dim: int):
        super().__init__()
        self.net = nn.Linear(S, latent_dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, z, s):
        return z * (1.0 + self.net(s))


class ResponseHead(nn.Module):
    """
    Auxiliary target (--response_weight): the firm's fingerprint change over the next h quarters predicted from its
    embedding and the realised macro scenario, Δf̂ = A z + Σ_m s_m B_m z, so that z has to carry how the firm responds
    to the macro (its exposures). Trained with the encoder; not used to score the embedding.
    """

    def __init__(self, latent_dim: int, S: int, K: int):
        super().__init__()
        self.lin = nn.Linear(latent_dim * (1 + S), K)

    def forward(self, z, s):
        return self.lin(torch.cat([z, (z[:, :, None] * s[:, None, :]).flatten(1)], dim=1))
