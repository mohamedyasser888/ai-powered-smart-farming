import zipfile
import json
import os
import shutil

keras_file = "data/final version model lstm.keras"
backup_file = "data/final version model lstm.keras.bak"
if not os.path.exists(backup_file):
    shutil.copy(keras_file, backup_file)

with zipfile.ZipFile(keras_file, 'r') as z:
    z.extractall("tmp_keras")

with open("tmp_keras/config.json", "r") as f:
    config_data = f.read()

# Replace DTypePolicy object with simple "float32" string
# The issue is Keras 3.3+ sometimes serializes dtype as a dict.
import re
# We can just parse JSON, walk the tree, and replace any dict that has class_name DTypePolicy
def fix_dtypes(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "DTypePolicy" and "config" in obj:
            return obj["config"].get("name", "float32")
        for k, v in obj.items():
            obj[k] = fix_dtypes(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = fix_dtypes(obj[i])
    return obj

cfg = json.loads(config_data)
cfg = fix_dtypes(cfg)

with open("tmp_keras/config.json", "w") as f:
    json.dump(cfg, f)

# Re-zip
with zipfile.ZipFile(keras_file, 'w') as z:
    for root, dirs, files in os.walk("tmp_keras"):
        for file in files:
            z.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), "tmp_keras"))

shutil.rmtree("tmp_keras")
print("Successfully patched config.json inside the .keras file!")
