import hashlib


# Compute MD5 hash of a file. Used to detect if PDFs have been modified
# and need re-embedding.
def get_file_hash(path):
    hashMd5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4098), b""):
            hashMd5.update(chunk)
    return hashMd5.hexdigest()
