import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / 'real_datasets' / 'xrf'


def load_dataset_config(dataset_name: str) -> dict:
    """Load and validate a dataset JSON config by name."""
    config_path = CONFIGS_DIR / f'{dataset_name}.json'
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found for dataset '{dataset_name}' at {config_path}"
        )
    with open(config_path) as f:
        return json.load(f)


def list_available_datasets() -> list[str]:
    """List all datasets that have JSON configs."""
    return sorted(p.stem for p in CONFIGS_DIR.glob('*.json'))


def get_compatible_datasets(model_name: str) -> list[str]:
    """List datasets compatible with a given model."""
    result = []
    for ds in list_available_datasets():
        config = load_dataset_config(ds)
        if model_name in config.get('compatible_models', []):
            result.append(ds)
    return result
