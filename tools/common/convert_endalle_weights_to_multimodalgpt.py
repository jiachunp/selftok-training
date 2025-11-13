import sys

sys.path.append(".")
import torch
from mimogpt.models.mimo.multi_modal_gpt import MultiModalGPT
from mimogpt.models.mimo.mimo_dalle import DalleModel


if __name__ == "__main__":
    # python -m my_scripts.convert_endalle_weights_to_multimodalgpt
    # old_weights_file = "/mnt/c/Users/z00451797/Downloads/endalle_f20_303965.pth"
    old_weights_file = "/ssd/ssd0/wny/ckpts/endalle/iter_315656_baseline_freeze20.pth"
    old_states = torch.load(old_weights_file, "cpu")

    config = dict(
        hf_version="v3",
        description="English Dalle",
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=False,
            vocab_size=30522 + 128,
            image_vocab_size=8192,
            gradient_checkpointing=8,
        ),
    )
    model_old = DalleModel(device="cpu", hf_version=config["hf_version"], **config["model_params"])
    model_old.load_state_dict(old_states)
    model_old.eval()

    old_states.pop("transformer.row_mask")
    old_states.pop("transformer.col_mask")
    old_states.pop("transformer.conv_mask")

    cfg = dict(
        n_layer=24,
        n_embd=2048,
        n_head=16,
        dropout=0.1,
        img_seq_length=1024,
        max_img_num=1,
        text_seq_length=128,
        cogview_sandwich_layernorm=True,
        text_vocab_size=30522 + 128,
        img_vocab_size=8192,
        use_checkpoint=True,
    )
    model = MultiModalGPT(**cfg)
    model.load_state_dict(old_states)
    model.eval()

    text = torch.randint(0, 100, (1, 128))
    img = torch.randint(0, 100, (1, 1024))

    out1, _ = model_old(text, img, cond_drop_prob=0.0)
    out2 = model(text, img, cond_drop_prob=0.0)
    diff = (out1 - out2).abs()
    # 6.0558e-05, 9.7729e-07
    print(diff.max(), diff.mean())
