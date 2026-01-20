import json

# Path to original StereoSet test file
INPUT_PATH = "data/stereoset/test.json"
OUTPUT_PATH = "data/stereoset/test_small.json"

# Load the dataset
with open(INPUT_PATH, "r") as f:
    data = json.load(f)

# Extract only 10 intrasentence examples
data["data"]["intrasentence"] = data["data"]["intrasentence"][:10]

# Save the new smaller dataset
with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ Successfully saved {len(data['data']['intrasentence'])} examples to {OUTPUT_PATH}")
