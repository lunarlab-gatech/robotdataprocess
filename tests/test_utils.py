import os
import urllib.request

def safe_urlretrieve(url, dest_path):
    """
    A method that retrives a file from a url, making sure
    to create the destination path if it doesn't exist.

    Throws:
        RuntimeError - If the recieved content is html.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=10) as response:
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            raise RuntimeError(f"Failed to download {url}: received HTML instead of the expected file")
        with open(dest_path, 'wb') as out_file:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)