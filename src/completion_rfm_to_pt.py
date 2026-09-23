"""
Converts a completion_rfm.py output (.pkl, 1-D direction vectors per
layer) into the SAME .pt container format train_sequence_rfm.py
produces -- the format load_directions_for_k(path, k=k) is already
known to read successfully (used against bag_of_words_rfm_directions.pt
and sequence_rfm_directions.pt in the existing controller pipelines).

*** UNVERIFIED: I have not seen train_controller.py / load_directions_for_k
itself, only its usage against train_sequence_rfm.py-produced files. This
mirrors that container shape as closely as possible, but I can't
guarantee it's exactly what load_directions_for_k expects. Sanity-check
before trusting a real run -- e.g.:

    python -c "
    from train_controller import load_directions_for_k
    d = load_directions_for_k('completion_rfm_directions.pt', k=1)
    print({layer: v.shape for layer, v in d.items()})
    "

and confirm the shapes look like what the sequence-RFM / bag-of-words-RFM
versions produce for the same call.
"""
import argparse
import pickle

import torch


def convert(pkl_path, pt_path, k=1):
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    directions_per_layer_1d = result["directions_per_layer"]  # {layer: (d,) tensor}
    layers = result.get("layers", list(directions_per_layer_1d.keys()))

    directions_per_layer_2d = {}
    for layer, vec in directions_per_layer_1d.items():
        vec = vec.detach().cpu() if hasattr(vec, "detach") else torch.as_tensor(vec)
        if vec.dim() == 1:
            vec = vec.unsqueeze(1)  # (d,) -> (d, 1), matching train_sequence_rfm.py's (d, k)
        elif vec.dim() != 2:
            raise ValueError(f"layer {layer}: unexpected direction shape {tuple(vec.shape)}")
        directions_per_layer_2d[layer] = vec

    torch.save({
        "directions_per_layer": directions_per_layer_2d,
        "layers": layers,
        "k": k,
        "source": "completion_rfm",
        "sign": result.get("sign"),
    }, pt_path)
    print(f"Wrote {pt_path} ({len(directions_per_layer_2d)} layers, "
          f"shape per layer e.g. {next(iter(directions_per_layer_2d.values())).shape})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_path", required=True, help="output of completion_rfm.py's --out_path")
    p.add_argument("--pt_path", required=True)
    p.add_argument("--k", type=int, default=1)
    args = p.parse_args()

    convert(args.pkl_path, args.pt_path, args.k)
