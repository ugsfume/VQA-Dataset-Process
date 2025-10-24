'''
This script scans for samples that are missing analysis_gt.json
Execute from the defect class directory
'''
import os

missing = []
present_count = 0

for root, dirs, files in os.walk('.'):
    # Skip hidden directories
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    # Only consider directories, not the root itself
    if root == '.':
        continue
    # Skip hidden directories at this level too
    if any(part.startswith('.') for part in os.path.relpath(root).split(os.sep)):
        continue
    if "analysis_gt.json" in files:
        present_count += 1
    else:
        missing.append(os.path.abspath(root))

for d in missing:
    print(d)
print(f"❌ Total subdirectories missing 'analysis_gt.json': {len(missing)}")
print(f"✅ Total subdirectories containing 'analysis_gt.json': {present_count}")
