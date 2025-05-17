from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import os
import io
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("framescope")

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="FrameScope API", description="API for analyzing webcam frames with Google Gemini")

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini API
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    logger.info("Gemini API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {e}")
    raise

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "FrameScope API is running"}

@app.post("/analyze")
async def analyze_frame(
    image: UploadFile = File(...),
    prompt: str = Form(...),
):
    """
    Analyze a webcam frame using Google Gemini API.
    
    Args:
        image: The webcam frame image
        prompt: The user's prompt for analysis
    
    Returns:
        Analysis result from Gemini
    """
    try:
        # Log the incoming request (without image content)
        logger.info(f"Received analyze request with prompt: {prompt}")
        
        # Read and validate the image
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Process the image using PIL
        try:
            img = Image.open(io.BytesIO(image_content))
            # Optionally resize to optimize for Gemini API
            img = img.resize((768, 576), Image.LANCZOS)
            
            # Convert back to bytes for Gemini API
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Prepare the prompt for Gemini
        enhanced_prompt = f"""
        Analyze this webcam frame and respond to the following request:
        
        User request: {prompt}
        
        Focus on describing what you see in the frame that's relevant to the user's request.
        Keep your response concise (2-3 sentences maximum).
        """
        
        # Send to Gemini API
        response = model.generate_content(
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": enhanced_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode('utf-8')}}
                    ]
                }
            ],
            generation_config={
                "max_output_tokens": 150,
                "temperature": 0.2,
            }
        )
        
        # Process and return the response
        result = response.text
        logger.info(f"Gemini response: {result}")
        return {"result": result}
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)