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
import time
import requests

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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY environment variable is not set")

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Define models to try in order of preference
MODELS = [
    "gemini-1.5-flash",  # Less resource-intensive than pro
    "gemini-1.5-pro",    # Original model
    "gemini-1.0-pro"     # Fallback option
]

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Rate limiting configuration for continuous monitoring
RATE_LIMIT_WINDOW = 60  # seconds
MAX_REQUESTS_PER_WINDOW = 20  # Adjust based on your API quota

# Simple rate limiter for continuous monitoring
class RateLimiter:
    def __init__(self, max_requests, window_size):
        self.max_requests = max_requests
        self.window_size = window_size
        self.requests = []
    
    def can_process(self):
        # Clean up old requests
        current_time = time.time()
        self.requests = [req_time for req_time in self.requests 
                         if current_time - req_time < self.window_size]
        
        # Check if we can process more
        if len(self.requests) < self.max_requests:
            self.requests.append(current_time)
            return True
        return False

# Initialize rate limiter
rate_limiter = RateLimiter(MAX_REQUESTS_PER_WINDOW, RATE_LIMIT_WINDOW)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "FrameScope API is running"}

async def generate_with_retry(model_name, contents, max_retries=MAX_RETRIES):
    """
    Helper function to generate content with retry logic
    """
    retries = 0
    last_error = None
    
    while retries < max_retries:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                contents=contents,
                generation_config={
                    "max_output_tokens": 150,
                    "temperature": 0.2,
                }
            )
            return response.text
        except Exception as e:
            last_error = e
            if "429" in str(e):  # Rate limit error
                retry_seconds = min(RETRY_DELAY * (2 ** retries), 30)  # Exponential backoff
                logger.warning(f"Rate limit hit for {model_name}. Retrying in {retry_seconds}s")
                await asyncio.sleep(retry_seconds)
                retries += 1
            else:
                # For non-rate-limit errors, don't retry
                raise e
    
    # If we get here, we've exhausted our retries
    logger.error(f"Failed after {max_retries} retries with model {model_name}")
    raise last_error

@app.post("/analyze")
async def analyze_frame(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    continuous: bool = Form(False),  # New parameter to indicate continuous mode
):
    """
    Analyze a webcam frame using Google Gemini API.
    
    Args:
        image: The webcam frame image
        prompt: The user's prompt for analysis
        continuous: Whether this is part of continuous monitoring
    
    Returns:
        Analysis result from Gemini
    """
    try:
        # Apply rate limiting for continuous requests
        if continuous and not rate_limiter.can_process():
            logger.warning("Rate limit applied to continuous monitoring request")
            return {
                "result": "Skipping analysis due to rate limiting",
                "status": "rate_limited"
            }
        
        # Log the incoming request (without image content)
        logger.info(f"Received analyze request with prompt: {prompt}, continuous: {continuous}")
        
        # Read and validate the image
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Process the image using PIL to optimize size
        try:
            img = Image.open(io.BytesIO(image_content))
            # Resize to optimize for Gemini API - smaller size to reduce token count
            img = img.resize((640, 480), Image.LANCZOS)
            
            # Convert back to bytes for Gemini API
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=85)  # Reduced quality to save tokens
            img_bytes = img_byte_arr.getvalue()
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Prepare the prompt for Gemini - keep it concise to reduce tokens
        enhanced_prompt = f"Analyze this image and respond to: {prompt}. Keep your response under 3 sentences."
        
        # Create the content structure
        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": enhanced_prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode('utf-8')}}
                ]
            }
        ]
        
        # Try each model in sequence until one works
        last_error = None
        for model_name in MODELS:
            try:
                logger.info(f"Attempting to use model: {model_name}")
                result = await generate_with_retry(model_name, contents)
                logger.info(f"Successfully analyzed with {model_name}")
                return {"result": result, "status": "success"}
            except Exception as e:
                logger.warning(f"Failed with model {model_name}: {str(e)}")
                last_error = e
        
        # If all models failed, return a generic error message
        if last_error:
            logger.error(f"All models failed. Last error: {last_error}")
            return {
                "result": "I'm currently experiencing high demand. Please try again in a few moments.",
                "status": "error"
            }
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        error_msg = str(e)
        if "429" in error_msg:
            return {
                "result": "Service is currently at capacity. Please try again in a few moments.",
                "status": "rate_limited"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)