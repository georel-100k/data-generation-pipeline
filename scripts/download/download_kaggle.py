import kagglehub

# Download latest version
path = kagglehub.dataset_download("imt2022118/abo-small")

print("Path to dataset files:", path)
