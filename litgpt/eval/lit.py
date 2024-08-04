from lm_eval.models.huggingface import HFLM
from litgpt.config import Config
from litgpt.model import GPT
from litgpt.tokenizer import Tokenizer
from litgpt.utils import load_checkpoint
import lightning as L
from transformers import PreTrainedTokenizerFast

from transformers import PreTrainedTokenizer

import torch

class WrappedGPT:
    def __init__(self, model, config, device):
        self.model = model
        self.device = device
        self.config = config

    def reset_kv_cache(self):
        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()

class LitLM(HFLM):
    def __init__(self, checkpoint_dir):
        checkpoint_path = checkpoint_dir / "lit_model.pth"

        tokenizer = Tokenizer(checkpoint_dir)
        assert tokenizer.backend == "huggingface"
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer.processor)

        precision = None
        self.fabric = L.Fabric(devices=1, precision=precision)
        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        with self.fabric.init_module(empty_init=True):
            model = GPT(config)
        with self.fabric.init_tensor():
            # model.max_seq_length = max_seq_length
            model.set_kv_cache(batch_size=1)
        model.eval()

        model = self.fabric.setup_module(model)

        load_checkpoint(self.fabric, model, checkpoint_path)
        
        device = self.fabric.device

        super().__init__(pretrained=WrappedGPT(model, config, device), tokenizer=tokenizer, backend="causal")

    @torch.inference_mode()
    def _model_call(self, inps, attn_mask=None, labels=None):
        assert attn_mask is None
        assert labels is None
        logits = self.model.model(inps)
        self.model.reset_kv_cache()
        return logits
