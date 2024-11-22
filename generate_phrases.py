import sys
import random

def check_required_packages():
    """
    Check if required packages are installed and provide installation instructions if not.
    """
    missing_packages = []

    # Check for torch
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")

    # Check for tqdm
    try:
        import tqdm
    except ImportError:
        missing_packages.append("tqdm")

    if missing_packages:
        print("The following required packages are missing:")
        for package in missing_packages:
            print(f"- {package}")
        print("\nPlease install the missing packages using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr you can use the following command to install all required packages:")
        print("pip install torch tqdm")
        sys.exit(1)

# Run the package check at the beginning
check_required_packages()

# Import necessary libraries
import itertools
import math
import os
import torch
from tqdm import tqdm

def load_bip39_words(file_path):
    """
    Load BIP39 words from a file.
    
    Args:
    file_path (str): Path to the file containing BIP39 words.
    
    Returns:
    list: A list of BIP39 words.
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def check_cuda_availability():
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
    bool: True if CUDA is available, False otherwise.
    """
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available. GPU device found:")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. GPU acceleration will not be possible.")
    return cuda_available

def generate_combinations_batch(num_words, word_count, batch_size, start_index, device):
    """
    Generate a batch of word index combinations.
    
    Args:
    num_words (int): Number of words in each phrase.
    word_count (int): Total number of words in the BIP39 list.
    batch_size (int): Number of combinations to generate.
    start_index (int): Starting index for the batch.
    device (torch.device): Device to use for tensor operations.
    
    Returns:
    torch.Tensor: A tensor of word index combinations.
    """
    batch = torch.zeros((batch_size, num_words), dtype=torch.long, device=device)
    
    for i in range(batch_size):
        idx = start_index + i
        for j in range(num_words):
            batch[i, j] = idx % word_count
            idx //= word_count
    
    return batch

def get_last_generated_index():
    """
    Find the index of the last generated phrase by checking existing files.
    
    Returns:
    int: The index of the last generated phrase, or 0 if no phrases have been generated.
    """
    files = [f for f in os.listdir('.') if f.startswith('BIP39_') and f.endswith('.txt')]
    if not files:
        return 0
    
    last_file = max(files)
    with open(last_file, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    
    words = last_line.strip().split()
    last_index = sum(words.index(word) * (2048 ** i) for i, word in enumerate(reversed(words)))
    return last_index + 1

def generate_random_phrase(words, num_words, generated_phrases):
    """
    Generate a random unique phrase.
    
    Args:
    words (list): List of BIP39 words.
    num_words (int): Number of words in each phrase.
    generated_phrases (set): Set of already generated phrases.
    
    Returns:
    str: A unique random phrase.
    """
    while True:
        phrase = ' '.join(random.choice(words) for _ in range(num_words))
        if phrase not in generated_phrases:
            generated_phrases.add(phrase)
            return phrase

def generate_phrases(words, num_words, total_combinations, use_cuda=False, is_random=False):
    """
    Generate BIP39 phrases and write them to files.
    
    Args:
    words (list): List of BIP39 words.
    num_words (int): Number of words in each phrase.
    total_combinations (int): Total number of combinations to generate.
    use_cuda (bool): Whether to use CUDA for GPU acceleration.
    is_random (bool): Whether to generate random phrases or alphabetical.
    """
    try:
        device = torch.device("cuda" if use_cuda else "cpu")
        phrases_per_file = 100_000_000
        phrases_per_cycle = 100_000_000_000
        word_count = len(words)
        
        global_start_index = get_last_generated_index() if not is_random else 0
        file_index = (global_start_index // phrases_per_file) + 1

        print(f"Starting generation from index {global_start_index}")

        generated_phrases = set()

        while global_start_index < total_combinations:
            cycle_end = min(global_start_index + phrases_per_cycle, total_combinations)
            
            for start_index in range(global_start_index, cycle_end, phrases_per_file):
                file_name = f"BIP39_{file_index:05d}.txt"
                print(f"Generating phrases in {file_name}...")
                
                with open(file_name, 'a') as current_file:
                    batch_size = min(phrases_per_file, cycle_end - start_index)
                    
                    with tqdm(total=batch_size, unit="phrase") as pbar:
                        if is_random:
                            for _ in range(batch_size):
                                phrase = generate_random_phrase(words, num_words, generated_phrases)
                                current_file.write(f"{phrase}\n")
                                pbar.update(1)
                        else:
                            if use_cuda:
                                combinations = generate_combinations_batch(num_words, word_count, batch_size, start_index, device)
                                for combo in combinations:
                                    phrase = ' '.join(words[idx.item()] for idx in combo)
                                    current_file.write(f"{phrase}\n")
                                    pbar.update(1)
                            else:
                                for i in range(batch_size):
                                    idx = start_index + i
                                    combo = []
                                    for _ in range(num_words):
                                        combo.append(idx % word_count)
                                        idx //= word_count
                                    phrase = ' '.join(words[idx] for idx in reversed(combo))
                                    current_file.write(f"{phrase}\n")
                                    pbar.update(1)
                
                print(f"Completed file {file_name}")
                file_index += 1
            
            global_start_index = cycle_end
            
            if global_start_index < total_combinations:
                print(f"\nGenerated {global_start_index} out of {total_combinations} phrases.")

        print("Phrase generation completed.")
    except Exception as e:
        print(f"An error occurred during phrase generation: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Ensure you have the latest NVIDIA GPU drivers installed.")
        print("2. Make sure you have installed PyTorch with CUDA support. You can do this by running:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   (Replace 'cu118' with your CUDA version if different)")
        print("3. Check if CUDA is properly set up by running:")
        print("   python -c 'import torch; print(torch.cuda.is_available())'")
        print("   This should print 'True' if CUDA is available.")
        print("4. If the issue persists, try running the script with CPU mode by answering 'n' when prompted for GPU usage.")
        print("5. If you continue to experience problems, please check your GPU compatibility and CUDA installation.")

if __name__ == "__main__":
    bip39_words = load_bip39_words('BIP39.txt')
    num_words = 12
    total_combinations = 2048**11 * 2048**(11-4)
    
    cuda_available = check_cuda_availability()
    use_cuda = False
    
    if cuda_available:
        choice = input("CUDA is available. Would you like to use GPU for faster processing? (y/n): ").lower()
        use_cuda = choice == 'y'
    
    generation_type = input("Would you like the results to be random or alphabetical? (r/a): ").lower()
    is_random = generation_type == 'r'

    if is_random:
        print("Generating random, unique phrases...")
    else:
        print(f"Generating {total_combinations} phrases alphabetically using {'GPU' if use_cuda else 'CPU'}...")
    
    generate_phrases(bip39_words, num_words, total_combinations, use_cuda, is_random)