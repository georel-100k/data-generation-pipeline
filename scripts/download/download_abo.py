import requests
from concurrent.futures import ThreadPoolExecutor
import os

URL = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
OUTPUT = "abo-3dmodels.tar"
NUM_THREADS = 16

def get_file_size(url):
    r = requests.head(url)
    return int(r.headers.get("Content-Length", 0))

def download_chunk(start, end, idx):
    headers = {"Range": f"bytes={start}-{end}"}
    response = requests.get(URL, headers=headers, stream=True)
    
    with open(f"part_{idx}", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def merge_files(num_parts):
    with open(OUTPUT, "wb") as outfile:
        for i in range(num_parts):
            with open(f"part_{i}", "rb") as infile:
                outfile.write(infile.read())
            os.remove(f"part_{i}")

def main():
    file_size = get_file_size(URL)
    chunk_size = file_size // NUM_THREADS

    ranges = []
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = start + chunk_size - 1 if i < NUM_THREADS - 1 else file_size - 1
        ranges.append((start, end, i))

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        executor.map(lambda x: download_chunk(*x), ranges)

    merge_files(NUM_THREADS)
    print("Download complete!")

if __name__ == "__main__":
    main()