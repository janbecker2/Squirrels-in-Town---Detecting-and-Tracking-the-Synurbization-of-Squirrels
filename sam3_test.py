from huggingface_hub import login 
import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from transformers import Sam3TrackerVideoProcessor, Sam3TrackerVideoModel
from PIL import Image 
import torch 
import numpy as np
from dotenv import load_dotenv
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
load_dotenv()

# Replace with your token from Step 2  
tokenValue = os.getenv("API_TOKEN")

login(token = tokenValue)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", device_map=device)

processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")


# def create_video_cutout(video_path, text_prompt, output_folder="frames"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load model with bfloat16 for your RTX 5060 speed boost
#     model = Sam3Model.from_pretrained("facebook/sam-3-hiera-large", torch_dtype=torch.bfloat16).to(device)
#     processor = Sam3Processor.from_pretrained("facebook/sam-3-hiera-large")

#     # 1. Initialize the video state (creates the memory bank)
#     # This prepares the model to track across frames
#     inference_state = model.init_state(video_path=video_path)

#     # 2. Use your text prompt to "find" the squirrel on frame 0
#     # This sets the initial target for the tracker
#     model.add_new_prompt(
#         inference_state=inference_state,
#         frame_idx=0,
#         obj_id=1,
#         text_prompt=text_prompt
#     )

#     # 3. Propagate (The Tracking Phase)
#     print(f"Tracking '{text_prompt}' across the video...")
    
#     # This loop goes through every frame automatically
#     for frame_idx, object_ids, mask_logits in model.propagate_in_video(inference_state):
#         # mask_logits contains the binary mask for the squirrel
#         mask = (mask_logits > 0.0).cpu().numpy()
        
#         # Here you can apply the mask to the frame to create your 'cutout'
#         # Example: save_frame_cutout(frame_idx, mask)
#         print(f"Processed frame {frame_idx}")

#     model.reset_state(inference_state)

# create_video_cutout(video_path = "F:/Outside3.mp4", text_prompt= "Squirrel")