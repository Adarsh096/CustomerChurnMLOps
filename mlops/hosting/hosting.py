from huggingface_hub import HfApi
import os

#common constants:
GITHUB_REPO_NAME = 'CustomerChurnMLOps'
HUGGINGFACE_SPACE_NAME = 'CustomerChurnMLOps'
HUGGINGFACE_DATASET_NAME = 'bank_customer_churn'
HUGGINGFACE_MODEL_NAME= 'churn-model'
HUGGINGFACE_USER_NAME = 'AdarshRL'

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="mlops/deployment",     # the local folder containing your files
    repo_id=f"{HUGGINGFACE_USER_NAME}/{HUGGINGFACE_SPACE_NAME}",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
