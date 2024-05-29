from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import src.python.easydel as ed


def main():
    config = AutoConfig.from_pretrained(
        "tiiuae/falcon-11B",
        trust_remote_code=True
    )

    setattr(config, "num_kv_heads", 8)
    setattr(config, "num_hidden_layers", 4)
    setattr(config, "num_attention_heads", 16)
    setattr(config, "max_position_embeddings", 128)
    setattr(config, "hidden_size", 128)
    setattr(config, "ffn_hidden_size", 256)
    setattr(config, "ff_factor", 2)
    print(config)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    print(model)

    model.save_pretrained("EasyDeL-Checkpoints/PT/Falcon-11BC")
    e_model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained("EasyDeL-Checkpoints/PT/Falcon-11BC")
    print(e_model)
    print(params)


if __name__ == "__main__":
    main()
