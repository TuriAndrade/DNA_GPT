from config.models import GPTConfig, GPTOptimConfig
from config.trainers import DNAGPTTrainerConfig
from config.dataloaders import LoadSeqDataConfig
from trainers.dna_gpt import DNAGPTTrainer
import torch

if __name__ == "__main__":
    trainer_config = DNAGPTTrainerConfig(
        model_config=GPTConfig.gpt_medium(
            vocab_size=6,
            block_size=512,
            attn_window=16,
        ),
        optim_config=GPTOptimConfig(),
        load_seq_data_config=LoadSeqDataConfig(
            file_path="/home/ubuntu/proj/data/dna_seq_human.txt",
            skip_first_line=True,
            seq_len=128,
            val_size=0.1,
            test_size=0.05,
            seed=1,
            dataset_name="Human DNA Dataset",
            seq_vocab=["A", "C", "G", "T", "N"],
            dataloader_config={
                "data_dtype": torch.long,
                "labels_dtype": torch.long,
                "batch_size": 8,
                "epoch_seed_mult": 1,
            },
        ),
        epochs=100,
        batch_size=8,
        save_path="./ARTIFACTS/DNA_GPT_5/",
        master_addr="localhost",
        master_port="1234",
        backend="nccl",
        main_device=0,
        process_timeout=10000,
    )

    trainer = DNAGPTTrainer(trainer_config)

    # trainer.spawn_train_ddp()

    trainer.evaluate_dummy_validation()
