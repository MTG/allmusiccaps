from huggingface_hub import create_repo
from huggingface_hub import HfApi, upload_folder, upload_file

folders = [
    "omar-rq-base-freesound-small",
]
for folder in folders:
    # repo_url = create_repo(f"mtg-upf/{folder}", private=False)

    api = HfApi()

    upload_folder(
        repo_id=f"mtg-upf/{folder}",
        folder_path=f"weights_light/{folder}",  # Current folder
        path_in_repo="",  # Upload everything at root
        repo_type="model",
    )

    upload_file(
        path_or_fileobj="MODELS_LICENSE",
        path_in_repo="LICENSE",
        repo_id=f"mtg-upf/{folder}",
        repo_type="model",
    )
