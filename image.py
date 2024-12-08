import os
import aiohttp
import replicate
from typing import Optional
import logging
import asyncio
from functools import partial
from status_manager import video_status
logger = logging.getLogger(__name__)

async def download_image(url: str, filename: str) -> Optional[str]:
    """Download image using aiohttp."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(filename, 'wb') as f:
                        f.write(await response.read())
                    return filename
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
    return None

async def generate_single_image(prompt: str, index: int, task_id: str) -> Optional[str]:
    """Generate a single image."""
    try:
        output_path = os.path.join("images", f"image_{task_id}_{index}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "9:16",
                "output_format": "png",
                "output_quality": 100,
                "num_inference_steps": 4
            }
        )
        
        if output and output[0]:
            downloaded_file = await download_image(output[0], output_path)
            if downloaded_file:
                return downloaded_file
        
        return None
        
    except Exception as e:
        logger.error(f"Error generating image {index}: {str(e)}")
        return None

async def generate_images_parallel(image_prompts: list[str], task_id: str) -> list[str]:
    """Generate images in parallel with better error handling."""
    tasks = []
    total_images = len(image_prompts)
    
    for i, prompt in enumerate(image_prompts):
        video_status.update_step("Generating Images", f"Creating image {i+1}/{total_images}")
        task = generate_single_image(prompt, i+1, task_id)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid_results = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to generate image {i+1}: {result}")
        elif result is None:
            logger.error(f"Failed to generate image {i+1}: No result")
        elif os.path.exists(result):
            valid_results.append(result)
        else:
            logger.error(f"Generated image file not found: {result}")
            
    if not valid_results:
        raise Exception("No valid images were generated")
        
    return valid_results