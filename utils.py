import json
from typing import Any, Callable, Dict

import requests
import tensorflow as tf


def get_activation(name: str) -> Callable:
    if name == "gelu":
        return lambda x: tf.keras.activations.gelu(x)

    raise ValueError(f"{name} not found")


def get_config(name: str) -> Dict[str, Any]:
    url = f"https://huggingface.co/{name}/resolve/main/config.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Cannot get config, status code: {response.status_code}, url: {url}")

    return json.loads(response.text)


def get_tokenizer_config(name: str):
    url = f"https://huggingface.co/{name}/resolve/main/tokenizer_config.json"
    response = requests.get(url)
    if response.status_code != 200:
        return {}

    return json.loads(response.text)
