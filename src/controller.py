"""
CLAS-style contextual controller for sequence-RFM steering subspaces
(pdf "Some first ideas on Sequence-Level RFM", Section 6).

We freeze the sequence-RFM steering subspace D_ell (shape (d, k),
already trained by train_sequence_rfm.py for a chosen k) and train a
small per-layer linear controller C_ell that replaces the fixed,
grid-searched global alpha with a per-token, per-layer coefficient:

    alpha_ell,t = C_ell([h_ell,t ; 1])              in R^k
    h'_ell,t    = h_ell,t + D_ell @ alpha_ell,t      in R^d

Following the pdf ("Can ignore here e_{y_i} and phi(t) as they do"),
the controller sees only [h_ell,t; 1] -- no label embedding, no
explicit position feature.

Training objective (pdf Eq. in Section 6), applied while freezing the
base model and D_ell:

    min_C  sum_i sum_t  -log p_theta(z_{i,t} | x_i, z_{i,<t}, y_i; D, C)

i.e. maximize likelihood of the model's OWN steered training
completions z_i under teacher forcing, while the steering hooks are
active on every layer simultaneously. All layers present in the
sequence-RFM directions file are trained JOINTLY in one backward pass
per batch (analogous to A-PSR's joint all-layer training in "Steer
Like the LLM") rather than fitting each layer's controller in
isolation -- this lets later layers compensate for what earlier layers
already did, and mirrors the all-layer convention your existing
steer_and_evaluate.py / persona_steer_and_evaluate.py already use at
inference time.

NOTE ON generation_utils.hook_model: that hook is inference-only
(designed to be called inside torch.no_grad() generation loops, per
steer_and_evaluate.py) and applies one FIXED direction with a global
scalar coefficient -- it doesn't (and structurally can't) support a
per-token, learned, subspace-valued coefficient with gradients flowing
back into a controller's parameters. So this file implements its own
forward-hook mechanism instead, matching hook_model's architecture
convention exactly (`model.model.layers[layer_idx]`, forward hook
returning a modified tuple when the block output is a tuple) so the
two are drop-in-compatible in every other respect. The eval-time hook
(SequenceRFMController.controlled_generate) is the differentiable-hook
analog of the old "hook_model + generate + clear_hooks" pattern, just
without a global alpha argument (the controller decides strength
itself, per token and per layer).

Assumed architecture convention: decoder-only, transformer blocks at
`model.model.layers[i]` (Llama-style), each returning a tuple whose
first element is the hidden-state tensor -- true for
Llama-3.1-8B-Instruct. If you're running a different architecture,
only `_get_decoder_layers` below needs to change.
"""
import torch
import torch.nn as nn


def _get_decoder_layers(language_model):
    """Returns the list-like module of transformer blocks, indexed by
    the same layer convention used elsewhere in this project (negative
    indices from the end, e.g. -1 = last layer), matching how
    sequence_rfm_directions.pt keys its `directions_per_layer` dict."""
    blocks = language_model.model.layers
    n = len(blocks)

    def get(layer_idx):
        # layer_idx may be negative (e.g. -1, -2, ...) or 0-indexed positive;
        # normalize the same way Python list indexing already does.
        return blocks[layer_idx]

    return get, n


class LayerController(nn.Module):
    """C_ell: a single-layer linear probe [h; 1] -> alpha in R^k."""

    def __init__(self, d_model, k, init_scale=1e-3):
        super().__init__()
        self.linear = nn.Linear(d_model, k, bias=True)
        # Small init so training starts close to "no steering" (alpha ~ 0)
        # rather than injecting a large random offset on step 0.
        nn.init.normal_(self.linear.weight, std=init_scale)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h):
        # h: (..., d_model) -> (..., k)
        return self.linear(h)


class SequenceRFMController(nn.Module):
    """
    Holds one frozen D_ell (d, k) and one trainable LayerController per
    layer, and manages the differentiable forward hooks that apply
    h'_ell,t = h_ell,t + D_ell @ alpha_ell,t during a forward pass.
    """

    def __init__(self, directions_per_layer, d_model, dtype=None):
        """
        directions_per_layer: {layer_idx: (d, k) tensor}, e.g. loaded
            from sequence_rfm_directions.pt's "directions_per_layer",
            for a FIXED k (see load_directions_for_k below).
        dtype: if None, inferred from the directions tensors themselves.
            Pass torch.bfloat16 (or the language_model's dtype) explicitly
            to guarantee the controller and frozen subspaces stay in the
            same dtype as the model -- required when the model was loaded
            with torch_dtype=torch.bfloat16 (e.g. Qwen2.5).
        """
        super().__init__()
        self.layers = sorted(directions_per_layer.keys())
        self.d_model = d_model
        self.k = next(iter(directions_per_layer.values())).shape[1]

        # Infer dtype from the stored directions when the caller doesn't
        # specify -- this makes the controller automatically match the
        # model dtype (bfloat16 for Qwen/Llama loaded with torch_dtype=bfloat16).
        if dtype is None:
            dtype = next(iter(directions_per_layer.values())).dtype
        self._dtype = dtype

        # Frozen subspaces, registered as buffers (not parameters) so
        # they move with .to(device) but never get gradients/optimizer
        # updates.
        self._D = {}
        for layer in self.layers:
            D = directions_per_layer[layer].to(dtype)
            buf_name = f"D_{self._safe_name(layer)}"
            self.register_buffer(buf_name, D)
            self._D[layer] = buf_name

        # Trainable per-layer controllers -- cast to the same dtype so
        # F.linear(hidden_bfloat16, weight_float32) no longer raises.
        self.controllers = nn.ModuleDict({
            self._safe_name(layer): LayerController(d_model, self.k).to(dtype)
            for layer in self.layers
        })

        self._active_hooks = []
        self._last_alphas = {}  # layer -> most recent alpha tensor (for logging)

    @staticmethod
    def _safe_name(layer_idx):
        return f"L{layer_idx}".replace("-", "neg")

    def get_D(self, layer):
        return getattr(self, self._D[layer])

    def get_controller(self, layer):
        return self.controllers[self._safe_name(layer)]

    def _make_hook(self, layer):
        D = self.get_D(layer)          # (d, k), frozen
        C = self.get_controller(layer)  # trainable

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            alpha = C(hidden)                    # (batch, seq, k)
            delta = alpha @ D.t()                # (batch, seq, d)
            steered = hidden + delta
            self._last_alphas[layer] = alpha.detach()

            if rest is not None:
                return (steered,) + rest
            return steered

        return hook

    def attach(self, language_model):
        """Registers forward hooks on every controlled layer. Call
        detach() when done (training step, or after a generation
        call) to avoid double-steering on the next forward pass."""
        get_layer, _ = _get_decoder_layers(language_model)
        for layer in self.layers:
            h = get_layer(layer).register_forward_hook(self._make_hook(layer))
            self._active_hooks.append(h)

    def detach(self):
        for h in self._active_hooks:
            h.remove()
        self._active_hooks = []

    def trainable_parameters(self):
        return self.controllers.parameters()

    # -- eval-time convenience, mirrors the old hook_model/generate/clear_hooks pattern --
    @torch.no_grad()
    def controlled_generate(self, language_model, tokenizer, formatted_prompt, max_new_tokens=100):
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(language_model.device)
        self.attach(language_model)
        try:
            output_ids = language_model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        finally:
            self.detach()
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)


def load_directions_for_k(seqrfm_path, k=None, dtype=None):
    """Loads a sequence_rfm_directions.pt (as produced by
    train_sequence_rfm.py, with or without --bag_of_words) and returns
    {layer: (d, k) tensor}.

    train_sequence_rfm.py already saves directions at whatever --k was
    passed at training time (D_ell = TopEigk(M_ell)), so this is mostly
    a thin loader -- but if k is passed here and is SMALLER than what
    the file was trained with, we truncate to the first k columns
    (they're already sorted by eigenvalue, largest first, so this is
    equivalent to having trained with the smaller k). Passing a k
    LARGER than what's stored raises, since we can't recover missing
    eigendirections after the fact -- retrain with a bigger --k instead.

    dtype: target dtype for the returned tensors.  If None, the tensors
        are returned in their stored dtype (typically float32 from
        eigendirection computation).  Pass torch.bfloat16 to match a
        model loaded with torch_dtype=bfloat16 -- the SequenceRFMController
        will also cast to this dtype automatically when dtype=None is
        passed to its constructor, so usually you don't need to set this
        explicitly.
    """
    data = torch.load(seqrfm_path)
    directions_per_layer = data["directions_per_layer"]
    out = {}
    for layer, D in directions_per_layer.items():
        # Preserve stored dtype unless the caller explicitly requests one.
        # Old code called .float() here which broke bfloat16 models.
        if dtype is not None:
            D = D.to(dtype)
        if k is not None:
            if D.shape[1] < k:
                raise ValueError(
                    f"{seqrfm_path} was trained with k={D.shape[1]}, "
                    f"cannot get k={k} from it -- retrain with --k {k}."
                )
            D = D[:, :k]
        out[layer] = D
    return out


def seqrfm_method_label(seqrfm_path):
    """Peeks at a sequence_rfm_directions.pt's saved metadata (written
    by train_sequence_rfm.py) to tell whether D_ell came from positional
    sequence-RFM (k_tau = RBF kernel) or the bag-of-words variant
    (k_tau === 1, position ignored) -- without loading/decoding the
    (small, but no need to touch) per-layer direction tensors.

    Used so downstream eval CSVs correctly label a controller trained on
    top of bag-of-words directions as "bag_of_words_rfm_controller"
    rather than defaulting to "sequence_rfm_controller" -- single source
    of truth is the directions file itself (what actually produced
    D_ell), not whatever a caller separately remembers passing.

    Old sequence_rfm_directions.pt files from before --bag_of_words
    existed lack this key entirely -- data.get(..., False) below
    correctly defaults those to "sequence_rfm", which is what they
    actually were (bag-of-words didn't exist yet)."""
    data = torch.load(seqrfm_path)
    return "bag_of_words_rfm" if data.get("bag_of_words", False) else "sequence_rfm"