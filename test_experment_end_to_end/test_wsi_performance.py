
import time
import numpy as np
import tifffile
import deap.gp as gp
from src import gp_setup, gp_ops, gp_context, gp_types
from src.gp_types import Image, Batch
import warnings
warnings.filterwarnings("ignore")
def load_wsi(path):
    print(f"Loading WSI from {path}...")
    # Read the image using tifffile
    # SVS files are typically multi-page TIFFs. Page 0 is usually the full resolution.
    with tifffile.TiffFile(path) as tif:
        # Check pages
        print(f"Found {len(tif.pages)} pages.")
        # Load the first page (highest resolution)
        # WARNING: This might be huge. 
        # For safety in this test environment, let's check dimensions.
        page = tif.pages[0]
        print(f"Dimensions: {page.shape}")
        
        # Check estimated size in memory
        est_size = np.prod(page.shape) # bytes (assuming uint8)
        print(f"Estimated size: {est_size / 1024**3:.2f} GB")
        
        if est_size > 2 * 1024**3: # Limit to 2GB
            print("Image too large for full load. Skipping Page 0.")
            if len(tif.pages) > 1:
                print("Attempting to load lower resolution (Page 1)...")
                page = tif.pages[1]
                print(f"Page 1 Dimensions: {page.shape}")
                return page.asarray()
            else:
                raise ValueError("Image too large and no other pages available.")

        # If it's extremely large, we might want to crop for the 'full' test 
        # to avoid OOM if the machine isn't powerful enough.
        # But the user asked to "run the image", so we'll try to load it.
        # If it fails, we catch it.
        try:
            img = page.asarray()
            print("Image loaded successfully.")
            return img
        except Exception as e:
            print(f"Failed to load full image: {e}")
            # Fallback to a smaller level if available or crop
            if len(tif.pages) > 1:
                print("Attempting to load lower resolution (Page 1)...")
                return tif.pages[1].asarray()
            else:
                raise

def prepare_image_batch(img_data):
    # img_data is (H, W, C) or (H, W)
    # We need to convert to [R_batch, G_batch, B_batch] where each is (N, H, W)
    # Here N=1
    
    if len(img_data.shape) == 2:
        # Grayscale
        h, w = img_data.shape
        # Treat as Gray
        # gp_ops expects list of channels. 
        # If we provide 3 channels, it works for RGB.
        # If we provide 1, we need to ensure GetGray works.
        # Let's replicate to 3 channels for simplicity if the tree expects RGB
        # Or better, provide 4 channels (R, G, B, Gray) as per gp_ops logic?
        # gp_ops._get_gray_iter checks if len > 3.
        
        # Let's just make it (1, H, W)
        data = img_data[np.newaxis, :, :]
        # We'll pass a list with just this, but we need to ensure the tree doesn't ask for Red if we don't have it.
        # But the primitive set has GetRed, GetGreen, GetBlue.
        # So we should probably provide 3 channels.
        return [data, data, data] # Fake RGB
        
    elif len(img_data.shape) == 3:
        # (H, W, C)
        h, w, c = img_data.shape
        # Transpose to (C, 1, H, W)
        data = img_data.transpose(2, 0, 1)[:, np.newaxis, :, :]
        # Convert to list of (1, H, W) arrays
        return [data[i] for i in range(c)]
    
    return []

def generate_random_tree():
    pset = gp_setup.create_primitive_set()
    # Generate a random tree
    # min_=2, max_=5 for depth
    expr = gp.genHalfAndHalf(pset, min_=2, max_=5)
    tree = gp.PrimitiveTree(expr)
    print(f"Generated Tree: {str(tree)}")
    return gp.compile(tree, pset)

def run_full_mode(img, tree_func):
    print("\n--- Running Full Mode ---")
    
    # Prepare data
    # We need at least 2 samples for classifiers to work in TRAIN mode.
    # So we duplicate the image.
    channels_single = prepare_image_batch(img)
    
    # Duplicate to make batch of 2
    # channels_single is [R, G, B] where each is (1, H, W)
    # We want [R, G, B] where each is (2, H, W)
    channels = []
    for c in channels_single:
        channels.append(np.concatenate([c, c], axis=0))
    
    # Setup Context
    gp_context.context.mode = gp_context.ExecutionMode.TRAIN
    # Dummy labels: 2 samples, classes 0 and 1
    gp_context.context.train_labels = np.array([0, 1]) 
    
    # Setup Input
    def iterator_factory():
        yield Batch(channels)
    
    image_wrapper = Image(iterator_factory)
    gp_ops.set_image(image_wrapper)
    
    # Run
    start_time = time.time()
    try:
        # tree_func is the compiled tree (Prediction object) if pset has no args
        if callable(tree_func):
            result_wrapper = tree_func()
        else:
            result_wrapper = tree_func
            
        # Consume iterator
        for batch in result_wrapper:
            pass
        end_time = time.time()
        print(f"Full Mode Execution Time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"Full Mode Failed: {e}")
        import traceback
        traceback.print_exc()

def run_patch_mode(img, tree_func, patch_size=(256, 256)):
    print(f"\n--- Running Patch Mode (Patch Size: {patch_size}) ---")
    
    h, w = img.shape[:2]
    ph, pw = patch_size
    
    # Create patches
    patches = []
    for i in range(0, h, ph):
        for j in range(0, w, pw):
            # Handle edge cases by cropping or padding? 
            # Let's just crop to valid patches for simplicity or take what's there
            patch = img[i:min(i+ph, h), j:min(j+pw, w)]
            # Pad if necessary to keep size constant? 
            # GP ops usually handle variable size, but batching usually requires same size.
            # If we put them in ONE batch, they must be same size.
            if patch.shape[0] == ph and patch.shape[1] == pw:
                patches.append(patch)
    
    print(f"Created {len(patches)} patches.")
    
    if not patches:
        print("No valid patches created.")
        return

    # Stack patches into a batch
    # (N, H, W, C)
    batch_data = np.stack(patches)
    
    # Prepare for gp_ops (List of (N, H, W))
    if len(batch_data.shape) == 4: # (N, H, W, C)
        # Transpose to (C, N, H, W)
        data_transposed = batch_data.transpose(3, 0, 1, 2)
        channels = [data_transposed[i] for i in range(data_transposed.shape[0])]
    else: # (N, H, W)
        data_expanded = batch_data[:, np.newaxis, :, :] # (N, 1, H, W) ?? No, (N, H, W) is fine for channel
        # But prepare_image_batch logic was for single image.
        # Here we have a batch.
        # Channel expects (N, H, W).
        channels = [batch_data] # Treat as single channel if gray
        if len(batch_data.shape) == 3: # (N, H, W) - Gray
             channels = [batch_data, batch_data, batch_data] # Fake RGB

    # Setup Context
    gp_context.context.mode = gp_context.ExecutionMode.TRAIN
    # Dummy labels
    gp_context.context.train_labels = np.random.randint(0, 2, len(patches))
    
    # Setup Input
    def iterator_factory():
        yield Batch(channels)
        
    image_wrapper = Image(iterator_factory)
    gp_ops.set_image(image_wrapper)
    
    # Run
    start_time = time.time()
    try:
        if callable(tree_func):
            result_wrapper = tree_func()
        else:
            result_wrapper = tree_func
            
        # Consume iterator
        for batch in result_wrapper:
            pass
        end_time = time.time()
        print(f"Patch Mode Execution Time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"Patch Mode Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    wsi_path = "data/TCGA-EW-A6SC-01Z-00-DX1.C2D50E7C-3AD0-4038-9A1C-CAE54A3FBE2F.svs"
    
    try:
        img = load_wsi(wsi_path)
        
        # Generate a tree
        tree_func = generate_random_tree()
        
        # Mode 1: Full
        run_full_mode(img, tree_func)
        
        # Mode 2: Patches
        run_patch_mode(img, tree_func)
        
    except Exception as e:
        print(f"Test Failed: {e}")
