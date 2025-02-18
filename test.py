
#%%
import torch
from PIL import Image
import mobileclip

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0',
        pretrained='./checkpoints/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')


#%%
# image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
# text = tokenizer(["a diagram", "a dog", "a cat"])

image = preprocess(Image.open("docs/AIPQ8933.JPG").convert('RGB')).unsqueeze(0)
text = tokenizer([
    "a person riding a bicycle on a sunny day",
    "a bowl of fresh fruit on a wooden table",
    "a person holding a compound bow",
    "a mountain landscape with a clear blue sky"
])

#%%
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)













# %% process local images and save embeddings

# %%
from torchvision import transforms

# Define augmentation pipeline
augmentation_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])










#%%
import os
import numpy as np
from PIL import Image
import torch
from pillow_heif import register_heif_opener  # Add HEIC support

# Register HEIC support with Pillow
register_heif_opener()

# Directory containing your images
# parent_dir = "/Volumes/Data/photos/20210807/HEIC"  # Parent folder containing all images
# parent_dir = "/Volumes/Data/photos/20211110"
parent_dirs = ["/Volumes/Data/photos/20211223", 
               "/Volumes/Data/photos/20220109", 
               "/Volumes/Data/photos/20220409", 
               "/Volumes/Data/photos/20220729", 
               "/Volumes/Data/photos/20221018", 
               "/Volumes/Data/photos/20221023", 
               "/Volumes/Data/photos/20221220", 
               "/Volumes/Data/photos/20230207", 
               "/Volumes/Data/photos/20230310", 
               "/Volumes/Data/photos/20230415", 
               "/Volumes/Data/photos/20230617", 
               "/Volumes/Data/photos/20230723", 
               "/Volumes/Data/photos/20230831", 
               "/Volumes/Data/photos/20230919", 
               "/Volumes/Data/photos/20231117_Roadtrip", 
               "/Volumes/Data/photos/20231231", 
               "/Volumes/Data/photos/20240326", 
               "/Volumes/Data/photos/20240505", 
               "/Volumes/Data/photos/20240615", 
               "/Volumes/Data/photos/20240702", 
               "/Volumes/Data/photos/pictures_from_chromebox_20221028", 
               "/Volumes/Data/photos/pictures_from_surface_20221028", 
               "/Volumes/Data/photos/SinceUS (20170909 - 20190201)"]
output_dir = "./embeddings"  # Directory to save embeddings
os.makedirs(output_dir, exist_ok=True)

# Function to process a single image
def process_image(image_path, relative_path):
    # Load the image
    max_retries = 3
    for attempt in range(max_retries):
        try:
            image = Image.open(image_path).convert('RGB')

            # Apply augmentations
            augmented_images = [augmentation_pipeline(image) for _ in range(5)]  # Generate 5 augmentations

            # Compute embeddings for the original and augmented images
            with torch.no_grad():
                original_embedding = model.encode_image(preprocess(image).unsqueeze(0))
                augmented_embeddings = [model.encode_image(img.unsqueeze(0)) for img in augmented_images]

            # Normalize embeddings
            original_embedding /= original_embedding.norm(dim=-1, keepdim=True)
            augmented_embeddings = [emb / emb.norm(dim=-1, keepdim=True) for emb in augmented_embeddings]

            # Save embeddings and metadata
            output_filename = os.path.splitext(image_path.replace("/", "_"))[0] + ".npz"
            output_path = os.path.join(output_dir, output_filename)
            np.savez(
                output_path,
                original_embedding=original_embedding.cpu().numpy(),
                augmented_embeddings=[emb.cpu().numpy() for emb in augmented_embeddings],
                filename=relative_path  # Save the relative path for reference
            )

            print(f"Processed and saved embeddings for {relative_path}")
            return True  # Success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {relative_path}: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
    print(f"Failed to process {relative_path} after {max_retries} attempts.")
    return False  # Failure

# Recursively process all images in the parent directory and its subfolders
for parent_dir in parent_dirs:
    for root, _, files in os.walk(parent_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".heic", ".heif")):
                # Get the full path of the image
                image_path = os.path.join(root, filename)
                
                # Get the relative path (relative to the parent directory)
                relative_path = os.path.relpath(image_path, parent_dir)
                
                # Process the image
                process_image(image_path, relative_path)
















#%% new process script, with try catch and continue
import os
import numpy as np
from PIL import Image
import torch
import time

# Directory containing your images
parent_dir = "/Volumes/Data/photos/"  # Parent folder containing all images
output_dir = "./embeddings"  # Directory to save embeddings
os.makedirs(output_dir, exist_ok=True)

# File to track the last processed folder
progress_file = os.path.join(output_dir, "progress.txt")

# Function to process a single image with retries
def process_image(image_path, relative_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')

            # Apply augmentations
            augmented_images = [augmentation_pipeline(image) for _ in range(5)]  # Generate 5 augmentations

            # Compute embeddings for the original and augmented images
            with torch.no_grad():
                original_embedding = model.encode_image(preprocess(image).unsqueeze(0))
                augmented_embeddings = [model.encode_image(img.unsqueeze(0)) for img in augmented_images]

            # Normalize embeddings
            original_embedding /= original_embedding.norm(dim=-1, keepdim=True)
            augmented_embeddings = [emb / emb.norm(dim=-1, keepdim=True) for emb in augmented_embeddings]

            # Save embeddings and metadata
            output_filename = os.path.splitext(relative_path.replace("/", "_"))[0] + ".npz"
            output_path = os.path.join(output_dir, output_filename)
            np.savez(
                output_path,
                original_embedding=original_embedding.cpu().numpy(),
                augmented_embeddings=[emb.cpu().numpy() for emb in augmented_embeddings],
                filename=relative_path  # Save the relative path for reference
            )

            print(f"Processed and saved embeddings for {relative_path}")
            return True  # Success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {relative_path}: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
    print(f"Failed to process {relative_path} after {max_retries} attempts.")
    return False  # Failure

# Function to get the last processed folder from the progress file
def get_last_processed_folder():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return f.read().strip()
    return None

# Function to save the last processed folder to the progress file
def save_last_processed_folder(folder):
    with open(progress_file, "w") as f:
        f.write(folder)

# Recursively process all images in the parent directory and its subfolders
last_processed_folder = get_last_processed_folder()
resume = last_processed_folder is not None

for root, _, files in os.walk(parent_dir):
    # Skip folders that have already been processed
    if resume:
        if root == last_processed_folder:
            resume = False  # Resume from this folder
        else:
            continue  # Skip already processed folders

    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            # Get the full path of the image
            image_path = os.path.join(root, filename)
            
            # Get the relative path (relative to the parent directory)
            relative_path = os.path.relpath(image_path, parent_dir)
            
            # Process the image
            if process_image(image_path, relative_path):
                # Save the last processed folder
                save_last_processed_folder(root)
            else:
                print(f"Skipping {relative_path} due to processing errors.")



























# %% search
from sklearn.metrics.pairwise import cosine_similarity

# Example query embedding (e.g., from text)
#%%

# query_embedding = model.encode_text(tokenizer(
#     ["a person holding a compound bow"])).cpu().numpy()
query_embedding = model.encode_text(tokenizer(
    ["a person holding a compound bow"])).detach().cpu().numpy()

# Load all embeddings and compute similarities
similarities = []
for filename in os.listdir(output_dir):
    if filename.endswith(".npz"):
        data = np.load(os.path.join(output_dir, filename))
        original_embedding = data["original_embedding"]
        augmented_embeddings = data["augmented_embeddings"]

        # Compute similarity with the original image
        sim = cosine_similarity(query_embedding, original_embedding)
        similarities.append((filename, sim))

        # Compute similarity with augmented images
        for aug_emb in augmented_embeddings:
            sim = cosine_similarity(query_embedding, aug_emb)
            similarities.append((filename, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# Print top results
for filename, sim in similarities[:5]:
    print(f"Filename: {filename}, Similarity: {sim}")
# %%
