"""
run_generative_only_inference_paperA.py -- Paper A Stage 0 copy of Paper B's
run_generative_only_inference.py. Two lines changed from the original,
both via sed on the server, recorded here for both sides to have the same
copy:

  1. CONFIG_FILE_PATH points at configs/inference_paperA_stage0.yaml
     (itself a copy of Paper B's configs/inference.yaml, with input_dir
     repointed at /data/pduttapramanik/rna_uncertainty/paperA/stage0/Inputs
     and samples_per_sequence changed from 10 to 50).
  2. output_dir is hardcoded to Paper A's own Predictions directory,
     instead of the original's hardcoded "Predictions_dropout_off" --
     that original hardcoding exists specifically so the dropout-off
     condition can never collide with Predictions/ (dropout-on); this
     copy keeps that same never-collide intent, just pointed at Paper A's
     own tree instead of Paper B's.

Everything else -- Sampler construction, dropout left OFF, the
list.txt-driven target loop, atom-mask handling -- is untouched from
Paper B's original and already validated there.

Usage (run from backbones/rnabpflow/, same as the original):
    python run_generative_only_inference_paperA.py
Target list comes from Inputs/list.txt under whatever input_dir the
config points at -- for this copy, that's
paperA/stage0/Inputs/list.txt, not Paper B's Inputs/list.txt.
"""
import os
import pickle

import yaml
from omegaconf import DictConfig

from inference import Sampler
from src.data.data_transform import make_atom_mask


def main():
    CONFIG_FILE_PATH = "configs/inference_paperA_stage0.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
        yaml_content = yaml.safe_load(f)

    input_dir = yaml_content["inference"]["input_dir"]
    # Fixed, distinct output directory -- Paper A's own tree, never
    # Paper B's Predictions/ or Predictions_dropout_off/, regardless of
    # what output_dir the config says.
    output_dir = "/data/pduttapramanik/rna_uncertainty/paperA/stage0/Predictions"
    yaml_content["inference"]["output_dir"] = output_dir
    cfg = DictConfig(yaml_content)

    sampler = Sampler(cfg)
    # Deliberately no enable_mc_dropout() call here. Sampler's own
    # construction already leaves the model in eval() mode, so all
    # nn.Dropout submodules stay OFF -- this is the entire point of
    # this condition.
    print("Dropout left in normal eval-mode (OFF). Only the model's "
          "own generative sampling stochasticity will vary across "
          "samples for the same target.")

    list_file_path = os.path.join(input_dir, "list.txt")
    with open(list_file_path, "r") as f:
        lines = f.readlines()

    id_list = []
    sample_count_list = []
    for line in lines:
        tokens = line.strip().split()
        id_list.append(tokens[0])
        if len(tokens) == 2:
            sample_count_list.append(int(tokens[1]))

    print(f"Running Condition A (generative-only) on {len(id_list)} targets, "
          f"writing to {output_dir}/")

    for idx, target_id in enumerate(id_list):
        if len(sample_count_list) == len(id_list):
            sampler._samples_cfg.samples_per_sequence = sample_count_list[idx]

        target_dir = os.path.join(output_dir, target_id)
        os.makedirs(target_dir, exist_ok=True)

        atom_dict = make_atom_mask(target_id, input_dir)
        pickle_file_path = os.path.join(target_dir, "map.pkl")
        with open(pickle_file_path, "wb") as f:
            pickle.dump(atom_dict, f)

        sampler.run_sampling(target_id)

        os.remove(pickle_file_path)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(id_list)} targets complete")

    print("Condition A (generative-only, dropout OFF) inference complete.")


if __name__ == "__main__":
    main()
