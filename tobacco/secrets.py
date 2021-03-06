import json

with open('/home/stephan/tobacco/code/tobacco_analytics/tobacco/secrets.json') as f:
    secrets = json.loads(f.read())

def get_secret(setting):
    """Get the secret variable or return explicit exception."""

    try:
        return secrets[setting]
    except KeyError:
        raise KeyError("Setting {} is not set in secrets.json.".format(setting))

