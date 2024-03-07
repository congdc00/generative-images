from huggingface_hub import HfApi

api = HfApi(endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="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")
api.upload_file(
    path_or_fileobj="data/model-house.zip",
    path_in_repo="model-house/part_01.zip",
    repo_id="congdc/thumb-youtube",
    repo_type="dataset",
)