import os

from omegaconf import OmegaConf

from src.guided_modality_adapter.model.model import SAASRModel


def main():
    config_path = os.path.join(
        os.getcwd(), "guided-modality-adapter/configs", "model.yaml"
    )
    config = OmegaConf.load(config_path)
    model = SAASRModel(config)
    print("Model initialized successfully.")


if __name__ == "__main__":
    main()
