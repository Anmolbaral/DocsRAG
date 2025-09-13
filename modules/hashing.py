import hashlib

def get_file_hash(path):
	hash_md5 = hashlib.md5()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(4098), b""):
			hash_md5.update(chunk)
	return hash_md5.hexdigest()



