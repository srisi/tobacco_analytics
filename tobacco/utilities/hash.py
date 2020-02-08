import hashlib
import json

def generate_hash(tuple):

    out = json.dumps(tuple).encode('utf-8')
    return hashlib.md5(out).hexdigest()


if __name__ == "__main__":
    generate_hash(('["addictive"]', [], [], [], ()))