import os
import pandas as pd
from glob import glob

# Map common names to scientific labels
scientific_name_map = {
    "ABBOTTS BABBLER": "Malacocincla abbotti",
    "ABBOTTS BOOBY": "Papasula abbotti",
    "ABYSSINIAN GROUND HORNBILL": "Bucorvus abyssinicus",
    "AFRICAN CROWNED CRANE": "Balearica regulorum",
    "AFRICAN EMERALD CUCKOO": "Chrysococcyx cupreus",
    "AFRICAN FIREFINCH": "Lagonosticta rubricata",
    "AFRICAN OYSTER CATCHER": "Haematopus moquini",
    "AFRICAN PIED HORNBILL": "Tockus fasciatus",
    "AFRICAN PYGMY GOOSE": "Nettapus auritus",
    "ALBATROSS": "Diomedea spp.",
    "ALBERTS TOWHEE": "Melozone aberti",
    "ALEXANDRINE PARAKEET": "Psittacula eupatria",
    "ALPINE CHOUGH": "Pyrrhocorax graculus",
    "ALTAMIRA YELLOWTHROAT": "Geothlypis flavovelata",
    "AMERICAN AVOCET": "Recurvirostra americana",
    "AMERICAN BITTERN": "Botaurus lentiginosus",
    "AMERICAN COOT": "Fulica americana",
    "AMERICAN FLAMINGO": "Phoenicopterus ruber",
    "AMERICAN GOLDFINCH": "Spinus tristis",
    "AMERICAN KESTREL": "Falco sparverius"
}

# Prepare output data
records = []
splits = ["train", "valid", "test"]
base_path = "../data"

for split in splits:
    split_path = os.path.join(base_path, split)
    if not os.path.exists(split_path):
        continue
    class_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

    for class_name in class_folders:
        class_path = os.path.join(split_path, class_name)
        images = glob(os.path.join(class_path, "*.jpg"))
        for img in images:
            records.append({
                "filepaths": os.path.relpath(img, base_path),
                "labels": class_name,
                "scientific label": scientific_name_map.get(class_name.upper(), ""),
                "data set": split,
                "class_id": class_folders.index(class_name)
            })

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("birds.csv", index=False)
print(f"✅ birds.csv created with {len(df)} entries.")
