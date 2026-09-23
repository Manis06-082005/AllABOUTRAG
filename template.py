import os


# Project folders
directories = [
    "data",
    "vectorstore"
]


# Project files
files = [
    "requirements.txt",
    ".env",
    "main.py"
]


# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")


# Create files
for file in files:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            pass

        print(f"Created file: {file}")


print("Project structure created successfully.")