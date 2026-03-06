"""
A helper script to download the FrodoBots 2k/8k datasets from AWS with parallel downloads and resume capability.
"""
import aiohttp
import asyncio
import aiofiles
import pandas as pd
import argparse
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Fast AWS dataset downloader with resume capability")
parser.add_argument("--save_dir", type=str, required=True, help="Directory where files will be saved")
parser.add_argument("--csv_file", type=str, default='https://frodobots-2k-dataset.s3.ap-southeast-1.amazonaws.com/complete-dataset.csv', help="Path to the CSV file containing dataset URLs")
parser.add_argument("--parallel_downloads", type=int, default=10, help="Number of parallel downloads")
args = parser.parse_args()

SAVE_DIR = Path(args.save_dir)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE = args.csv_file

# Load URLs from CSV file
def load_urls(csv_file):
    df = pd.read_csv(csv_file)
    return df['url'].tolist()

# Asynchronous file download with resume capability
async def download_file(session, url, save_path):
    chunk_size = 8192
    headers = {}
    file_mode = 'wb'
    existing_size = 0

    if save_path.exists():
        existing_size = save_path.stat().st_size
        headers['Range'] = f'bytes={existing_size}-'
        file_mode = 'ab'

    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with session.get(url, timeout=timeout, headers=headers) as response:
            if response.status in (200, 206):
                total_size = int(response.headers.get("content-length", 0)) + existing_size
                with tqdm_asyncio(total=total_size, initial=existing_size, unit="B", unit_scale=True, desc=save_path.name, leave=True) as progress_bar:
                    async with aiofiles.open(save_path, file_mode) as file:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            await file.write(chunk)
                            progress_bar.update(len(chunk))
            elif response.status == 416:
                print(f"File already fully downloaded: {save_path.name}")
            else:
                print(f"Failed to download {url} - Status: {response.status}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Main async function
async def main():
    urls = load_urls(CSV_FILE)
    semaphore = asyncio.Semaphore(args.parallel_downloads)
    connector = aiohttp.TCPConnector(keepalive_timeout=300)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def bounded_download(url):
            filename = Path(url).name
            save_path = SAVE_DIR / filename
            async with semaphore:
                await download_file(session, url, save_path)

        tasks = [bounded_download(url) for url in urls]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
