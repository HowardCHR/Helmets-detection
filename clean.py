import os

label_dir = 'data/test/labels'
keep_class = 0

for root, dirs, files in os.walk(label_dir):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id == keep_class:
                    new_lines.append(' '.join(parts) + '\n')
            with open(file_path, 'w') as f:
                f.writelines(new_lines)