from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
import os
import logging
from fastapi import WebSocket 
from fastapi import UploadFile
import shutil
from main import main
from status_manager import video_status
from starlette.responses import JSONResponse
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager
from typing import Dict, List
import json
import tempfile
from datetime import datetime
from pathlib import Path
from main import regenerate_video  # Add this at the top with other imports
from connection_manager import connection_manager
BASE_DIR = Path(__file__).parent.absolute()
# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path of the current directory
if os.environ.get('RENDER'):
    BASE_DIR = Path('/opt/render/project/src')
else:
    BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()
class VideoRequest(BaseModel):
    text: str
    caption_style: str
    image_style: str
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
DIRS = [
    "generated_videos",
    "generated_videos/thumbnails",
    "images",
    "videos",
    "audio",
    "static",
    "static/components"
]

for dir_name in DIRS:
    os.makedirs(BASE_DIR / dir_name, exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/generated_videos", StaticFiles(directory=str(BASE_DIR / "generated_videos")), name="generated_videos")
app.mount("/images", StaticFiles(directory=str(BASE_DIR / "images")), name="images")
app.mount("/videos", StaticFiles(directory=str(BASE_DIR / "videos")), name="videos")
app.mount("/audio", StaticFiles(directory=str(BASE_DIR / "audio")), name="audio")

async def serve_html(filename: str) -> HTMLResponse:
    """Helper function to serve HTML files"""
    try:
        file_path = BASE_DIR / filename
        logger.info(f"Attempting to serve {file_path}")
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error serving {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@app.head("/")
async def get_home(request: Request):
    """Serve the index page"""
    return await serve_html("index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
@app.get("/create")
async def get_create():
    """Serve the create page"""
    return await serve_html("create.html")

@app.get("/create.html")
async def get_create_html():
    """Alternative route for create.html"""
    return await serve_html("create.html")  # Add await here
@app.get("/projects")
async def get_projects():
    """Serve the projects page"""
    return await serve_html("projects.html")
@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await connection_manager.connect(task_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await connection_manager.disconnect(task_id)
@app.get("/projects.html")
async def get_projects_html():
    """Alternative route for projects.html"""
    return await serve_html("projects.html")

@app.get("/pricing")
async def get_pricing():
    """Serve the pricing page"""
    return await serve_html("pricing.html")

@app.get("/pricing.html")
async def get_pricing_html():
    """Alternative route for pricing.html"""
    return await serve_html("pricing.html")

# Serve static files like logo.png
@app.get("/logo.png")
async def get_logo():
    return FileResponse("static/logo.png")

def save_project_metadata(project_id: str, title: str):
    """Save project metadata to a JSON file"""
    project = {
        "id": project_id,
        "title": title,
        "created_at": datetime.now().isoformat()
    }
    
    projects_file = "generated_videos/projects.json"
    projects = []
    
    if os.path.exists(projects_file):
        with open(projects_file, "r") as f:
            try:
                projects = json.load(f)
            except json.JSONDecodeError:
                projects = []
    
    projects.append(project)
    
    with open(projects_file, "w") as f:
        json.dump(projects, f)



@app.get("/api/status")
async def get_status():
    """Get the current status of video creation"""
    try:
        return JSONResponse({
            "status": video_status.status,
            "progress": video_status.progress,
            "error": video_status.error,
            "current_step": video_status.current_step,
            "step_details": video_status.step_details
        })
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": "Internal server error",
                "progress": 0,
                "current_step": "Error",
                "step_details": str(e)
            }
        )
@app.get("/edit")
async def get_edit():
    """Serve the edit page"""
    return await serve_html("edit.html")

@app.get("/edit.html")
async def get_edit_html():
    """Alternative route for edit.html"""
    return await serve_html("edit.html")

@app.get("/api/video-metadata/{video_id}")
async def get_video_metadata(video_id: str):
    try:
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Video metadata not found")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Update image paths to include task_id
        for segment in metadata['segments']:
            segment['imagePath'] = f"/images/image_{video_id}_{segment['index']+1}.png"
            
        return metadata
    except Exception as e:
        logger.error(f"Error getting video metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/regenerate-segment")
async def regenerate_segment(request: dict):  # Changed Dict to dict
    """Regenerate a specific segment of the video with new prompt"""
    try:
        video_id = request.get('videoId')
        segment_index = request.get('segmentIndex')
        new_prompt = request.get('prompt')
        
        # Load metadata
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Update prompt in metadata
        metadata['segments'][segment_index]['prompt'] = new_prompt
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        # Generate new image and update video
        result = await main.regenerate_segment(
            video_id=video_id,
            segment_index=segment_index,
            new_prompt=new_prompt,
            metadata=metadata
        )
        
        if result.get('status') == 'success':
            return {"status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Failed to regenerate segment")
            
    except Exception as e:
        logger.error(f"Error regenerating segment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-segment")
async def update_segment(request: Dict):
    """Update a specific segment's properties (transition, style)"""
    try:
        video_id = request.get('videoId')
        segment_index = request.get('segmentIndex')
        updates = request.get('updates', {})
        
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Video metadata not found")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Update segment properties
        segment = metadata['segments'][segment_index]
        if 'transition' in updates:
            segment['transition'] = updates['transition']
        if 'style' in updates:
            segment['style'] = updates['style']
            
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error updating segment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/upload-segment-image/{video_id}/{segment_index}")
async def upload_segment_image(video_id: str, segment_index: int, file: UploadFile):
    """Handle custom image upload for a video segment"""
    try:
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Video not found")
            
        image_filename = f"image_{video_id}_{segment_index + 1}.png"
        image_path = os.path.join("images", image_filename)
        
        os.makedirs("images", exist_ok=True)
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Update metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        metadata['segments'][segment_index]['image_path'] = image_path
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return {"status": "success", "path": image_path}
        
    except Exception as e:
        logger.error(f"Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-captions")
async def update_captions(request: Dict):
    """Update caption properties (style, emoji)"""
    try:
        video_id = request.get('videoId')
        captions = request.get('captions', [])
        
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Video metadata not found")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        metadata['captions'] = captions
            
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error updating captions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-video")
async def update_video(request: Dict):
    try:
        video_id = request.get('videoId')
        segments = request.get('segments')
        
        logger.info(f"Updating video {video_id} with {len(segments)} segments")
        
        # Load metadata
        metadata_path = f"generated_videos/metadata_{video_id}.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Update metadata with new transitions and prompts
        for segment in segments:
            idx = segment['index']
            logger.info(f"Updating segment {idx} with transition: {segment.get('transition')}")
            metadata['segments'][idx]['transition'] = segment.get('transition', 'fade')
            metadata['segments'][idx]['prompt'] = segment.get('prompt', '')
            
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        # Regenerate video with new transitions
        try:
            result = await regenerate_video(
                video_id=video_id,
                metadata=metadata
            )
            logger.info(f"Regeneration result: {result}")
        except Exception as regen_error:
            logger.error(f"Video regeneration error: {str(regen_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Video regeneration failed: {str(regen_error)}"
            )

        if result and result.get('status') == 'success':
            return {"status": "success"}
        else:
            error_msg = f"Failed to update video. Result: {result}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except Exception as e:
        logger.error(f"Error in update_video: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to update video: {str(e)}"
        )

# server.py
async def create_video_task(text: str, caption_style: str, image_style: str, task_id: str):
    try:
        # Now connection_manager is properly imported
        await connection_manager.connect(task_id)
        video_status.status = "processing"
        video_status.progress = 0
        
        
        # Increase timeout to 1 hour
        async with asyncio.timeout(3600):  # 60 minutes timeout
            result = await main(
                story_input=text, 
                caption_style=caption_style,
                image_style=image_style,
                task_id=task_id
            )
            
            if result and result.get("status") == "success":
                # Generate a meaningful title from the first few words of the text
                words = text.split()
                title = " ".join(words[:5]) + "..." if len(words) > 5 else text
                
                # Save project metadata for the projects page
                save_project_metadata(
                    project_id=task_id,
                    title=title
                )
                
                # Ensure metadata includes caption style and final paths
                metadata_path = f"generated_videos/metadata_{task_id}.json"
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        metadata.update({
                            'caption_style': caption_style,
                            'final_video_path': result.get("video_path"),
                            'project_title': title,
                            'creation_date': datetime.now().isoformat()
                        })
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f)
                    except Exception as e:
                        logger.error(f"Error updating metadata file: {str(e)}")
                
                video_status.status = "complete"
                video_status.progress = 100
                logger.info(f"Video creation completed successfully: {result.get('video_path')}")
                
                # Verify that the video file exists
                video_path = f"generated_videos/video_{task_id}.mp4"
                if not os.path.exists(video_path):
                    raise Exception("Video file not found after creation")
                
                return result
            else:
                raise Exception("Video creation failed")
                
    except asyncio.TimeoutError:
        logger.error(f"Video creation timed out for task {task_id}")
        video_status.status = "error"
        video_status.error = "Video creation timed out after 30 minutes"
        raise
    except Exception as e:
        logger.error(f"Error in create_video_task: {str(e)}")
        video_status.status = "error"
        video_status.error = str(e)
        raise
    finally:
        await connection_manager.disconnect(task_id)
@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
        """Delete a project and its associated files"""
        try:
            # Delete video file
            video_path = f"generated_videos/video_{project_id}.mp4"
            if os.path.exists(video_path):
                os.remove(video_path)
            
            # Delete thumbnail if it exists
            thumbnail_path = f"generated_videos/thumbnails/{project_id}.jpg"
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
            
            # Remove project from projects.json
            projects_file = "generated_videos/projects.json"
            if os.path.exists(projects_file):
                with open(projects_file, "r") as f:
                    projects = json.load(f)
                
                # Filter out the deleted project
                projects = [p for p in projects if p["id"] != project_id]
                
                with open(projects_file, "w") as f:
                    json.dump(projects, f)
            
            return JSONResponse({"status": "success"})
        except Exception as e:
            logger.error(f"Error deleting project: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Failed to delete project: {str(e)}"
                }
            )
@app.post("/api/create-video")
async def create_video_endpoint(request: VideoRequest, background_tasks: BackgroundTasks):
    """Endpoint to create a new video"""
    try:
        logger.info(f"Received video creation request: {request}")
        
        # Reset status
        video_status.status = "starting"
        video_status.progress = 0
        video_status.error = None
        video_status.current_step = "Initializing"
        video_status.step_details = "Starting video creation"
        
        # Generate unique task ID
        task_id = os.urandom(16).hex()
        
        # Add task to background tasks
        background_tasks.add_task(
            create_video_task,
            request.text,
            request.caption_style,
            request.image_style,
            task_id
        )
        
        return JSONResponse({
            "status": "started",
            "task_id": task_id
        })
    except Exception as e:
        logger.error(f"Error starting video creation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )
@app.get("/generated_videos/{filename}")
async def serve_video(filename: str):
    """Serve video files from the generated_videos directory"""
    try:
        video_path = os.path.join("generated_videos", filename)
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=filename  # This will be used as the download filename
        )
    except Exception as e:
        logger.error(f"Error serving video {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/projects")
async def get_projects_list():
    """Get list of all projects"""
    try:
        projects_file = os.path.join("generated_videos", "projects.json")
        if not os.path.exists(projects_file):
            return []
        
        with open(projects_file, "r") as f:
            projects = json.load(f)
            
        # Filter out projects whose video files don't exist
        valid_projects = []
        for project in projects:
            video_path = os.path.join("generated_videos", f"video_{project['id']}.mp4")
            if os.path.exists(video_path):
                valid_projects.append(project)
        
        return valid_projects
    except Exception as e:
        logger.error(f"Error getting projects: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load projects: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",  # Change from 127.0.0.1 for Render
        port=port,       # Use PORT from environment
        timeout_keep_alive=3000,
        limit_concurrency=500,
        backlog=1000
    )
    
