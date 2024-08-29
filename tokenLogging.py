def log_in():
    from dotenv import load_dotenv
    import os
    from huggingface_hub import login
    import wandb
    # Load environment variables from .env file
    load_dotenv()

    # Access the Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    wb_token = os.getenv("WANDB_TOKEN")

    login(token=hf_token)

    wandb.login(key=wb_token)
    run = wandb.init(
        project='LoRA on Llama3.1 for Sentiment Analysis',
        job_type="training",
        anonymous="allow"
    )