from huggingface_hub import HfApi

api = HfApi(endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="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")
api.upload_folder(
    folder_path="data/model-house",
    path_in_repo="model-house/",
    repo_id="congdc/thumb-youtube",
    repo_type="dataset",
)