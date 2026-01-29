import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import asyncio
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware #Cross-Origin Resource Sharing.
from contextlib import asynccontextmanager
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel
import tensorflow as tf
from keras.applications import vgg16, resnet50
import numpy as np
import cv2
import io
import base64
import pydicom
from PIL import Image
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from datetime import datetime, timedelta
from bson import ObjectId
from dotenv import load_dotenv
import re
from huggingface_hub import hf_hub_download

# Import local utility from data_utils.py
try:
    from data_utils import letterbox
except ImportError:
    raise ImportError("Could not import 'letterbox' from 'data_utils.py'. Ensure the file exists in the same directory.")

# Load environment variables
load_dotenv()

def download_model_from_hf(repo_id, filename, save_path):
    """Downloads a model file from a Hugging Face Hub repository."""
    if os.path.exists(save_path):
        print(f"✅ {save_path} already exists.")
        return True
    
    print(f"⬇️ Downloading '{filename}' from Hugging Face repo '{repo_id}'...")
    try:
        # hf_hub_download will download the file and cache it.
        # To place it directly in our project folder, we specify local_dir.
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(save_path),
            local_dir_use_symlinks=False # Important for serverless environments
        )
        print(f"✨ {save_path} download successful!")
        return True
    except Exception as e:
        print(f"❌ Failed to download from Hugging Face. Error: {e}")
        return False

async def load_single_model(model_name: str):
    """
    Loads a single Keras model on-demand.
    Checks if the model file exists in the designated path, downloads it from
    Hugging Face if not, and then loads it into the global MODELS dictionary.
    This is designed for serverless environments where models are loaded lazily.
    """
    if model_name not in MODEL_CONFIG:
        print(f"⚠️ Attempted to load unknown model '{model_name}'")
        raise HTTPException(status_code=400, detail=f"Unknown model name: {model_name}")

    config = MODEL_CONFIG[model_name]
    model_path = config["path"]

    # In a serverless environment, the /tmp directory is writable.
    # Check if model file exists, if not, download it. This is crucial for cold starts.
    if not os.path.exists(model_path):
        hf_repo_id = os.getenv("HF_REPO_ID")
        if hf_repo_id:
            filename_in_repo = os.path.basename(model_path)
            print(f"⬇️ Model '{model_name}' not found at {model_path}. Downloading from Hugging Face...")
            try:
                # Run download in a separate thread to avoid blocking the event loop
                await asyncio.to_thread(
                    download_model_from_hf,
                    repo_id=hf_repo_id,
                    filename=filename_in_repo,
                    save_path=model_path
                )
            except Exception as e:
                print(f"❌ Failed to download model '{model_name}': {e}")
                raise HTTPException(status_code=503, detail=f"Could not download model '{model_name}'.")
        else:
            print(f"❌ Model file not found at {model_path} and HF_REPO_ID is not set.")
            raise HTTPException(status_code=503, detail=f"Model file for '{model_name}' is missing.")

    try:
        print(f"🧠 Loading model '{model_name}' into memory from {model_path}...")
        # Model loading is CPU/IO bound, so run it in a thread
        MODELS[model_name] = await asyncio.to_thread(tf.keras.models.load_model, model_path)
        print(f"✅ Model '{model_name}' loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading model '{model_name}': {e}")
        raise HTTPException(status_code=503, detail=f"Could not load model '{model_name}'.")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
db_client = None
history_collection = None
fs = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global db_client, history_collection, fs
    try:
        db_client = AsyncIOMotorClient(MONGO_URI)
        # Force a connection check to fail fast if DB is unreachable
        await db_client.admin.command('ping')
        history_collection = db_client.brain_tumor_db.history
        fs = AsyncIOMotorGridFSBucket(db_client.brain_tumor_db)
        print(f"✅ Connected to MongoDB at {MONGO_URI}")
    except Exception as e:
        print(f"⚠️ Could not connect to MongoDB at {MONGO_URI}: {e}")
    yield
    # Code to run on shutdown (optional)
    if db_client:
        db_client.close()
    print("--- Server shutting down ---")

app = FastAPI(
    title="Brain Tumor Detection API",
    description="Layer 1: Backend for Brain Tumor Detection. Handles Inference, DICOM processing, and Grad-CAM.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS to allow the Streamlit frontend (Layer 2) to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # It tells the browser, "I accept requests from any website or port."
    allow_credentials=True, # Allows the frontend to send cookies
    allow_methods=["*"],
    allow_headers=["*"], # Allows the frontend to send any type of data header (like Content-Type: application/json or custom headers).
)

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Brain Tumor Detection API is running. Go to /docs for API documentation."}

# Global Configuration
MODELS = {}
# For serverless environments like Vercel, /tmp is the only writable directory.
# We check for the 'VERCEL' environment variable to decide the save path.
MODEL_SAVE_DIR = "/tmp/models" if "VERCEL" in os.environ else os.path.dirname(os.path.abspath(__file__))
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_CONFIG = {
    "VGG16": {
        "path": os.path.join(MODEL_SAVE_DIR, os.path.basename(os.getenv("VGG16_MODEL_PATH", "VGG16_brain_tumor_detection_model.keras"))),
        "target_size": (224, 224),
        "preprocess_func": vgg16.preprocess_input
    },
    "ResNet50": {
        "path": os.path.join(MODEL_SAVE_DIR, os.path.basename(os.getenv("RESNET50_MODEL_PATH", "ResNet50_brain_tumor_detection_model.keras"))),
        "target_size": (224, 224),
        "preprocess_func": resnet50.preprocess_input
    }
}
# Configurable thresholds for the traffic light system
HIGH_CONFIDENCE_THRESHOLD = 0.70
LOW_CONFIDENCE_THRESHOLD = 0.30

# Authentication Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SUPER_SECRET_KEY_CHANGE_THIS_IN_PRODUCTION")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class User(BaseModel):
    username: str

class Token(BaseModel):
    access_token: str
    token_type: str
    is_admin: bool = False

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class DeleteAccountRequest(BaseModel):
    password: str

class UserRoleUpdate(BaseModel):
    is_admin: bool

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    # Check if user exists in DB
    user = await db_client.brain_tumor_db.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return User(username=user['username'])

async def get_full_user_from_token(token: str = Depends(oauth2_scheme)):
    """
    Dependency that decodes the JWT token, fetches the full user document
    from the database, and returns it. This is more efficient than re-fetching.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    user = await db_client.brain_tumor_db.users.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin_user(user: dict = Depends(get_full_user_from_token)):
    # Check if is_admin is True (bool) or "true" (string), case-insensitive
    if str(user.get("is_admin", False)).lower() != "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Admin privileges required"
        )
    return user

# Auth Endpoints
@app.post("/register")
async def register(
    username: str = Form(...),
    password: str = Form(...),
    recovery_key: str = Form(...)
):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    existing_user = await db_client.brain_tumor_db.users.find_one({"username": username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(password)
    hashed_recovery = get_password_hash(recovery_key)
    await db_client.brain_tumor_db.users.insert_one({
        "username": username, 
        "hashed_password": hashed_password,
        "recovery_key_hash": hashed_recovery
    })
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    user = await db_client.brain_tumor_db.users.find_one({"username": form_data.username})
    if user:
        print(f"DEBUG LOGIN: User '{form_data.username}' found. is_admin value: {user.get('is_admin')} (Type: {type(user.get('is_admin'))})")
    if not user or not verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user['username']})
    
    # Normalize is_admin to a boolean for the response
    is_admin_bool = str(user.get("is_admin", False)).lower() == "true"
    return {
        "access_token": access_token, 
        "token_type": "bearer", 
        "is_admin": is_admin_bool
    }

@app.post("/change-password")
async def change_password(password_data: PasswordChange, user: dict = Depends(get_full_user_from_token)):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    # The `user` dict is now passed directly from the dependency, no need to fetch again.
    if not verify_password(password_data.old_password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Incorrect old password")
    
    new_hashed_password = get_password_hash(password_data.new_password)
    await db_client.brain_tumor_db.users.update_one(
        {"username": user["username"]},
        {"$set": {"hashed_password": new_hashed_password}}
    )
    return {"message": "Password updated successfully"}

@app.post("/reset-password")
async def reset_password(
    username: str = Form(...),
    recovery_key: str = Form(...),
    new_password: str = Form(...)
):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")
        
    user = await db_client.brain_tumor_db.users.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    stored_recovery_hash = user.get("recovery_key_hash")
    if not stored_recovery_hash:
            raise HTTPException(status_code=400, detail="No recovery key set for this account")
            
    if not verify_password(recovery_key, stored_recovery_hash):
        raise HTTPException(status_code=400, detail="Invalid recovery key")
        
    new_hashed_password = get_password_hash(new_password)
    await db_client.brain_tumor_db.users.update_one(
        {"username": username},
        {"$set": {"hashed_password": new_hashed_password}}
    )
    return {"message": "Password reset successfully"}

@app.delete("/delete-account")
async def delete_account(
    request: DeleteAccountRequest, 
    user: dict = Depends(get_full_user_from_token)
):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    # Verify password
    if not verify_password(request.password, user['hashed_password']):
        raise HTTPException(status_code=400, detail="Invalid password")
    
    # 1. Find all history to delete files
    cursor = history_collection.find({"username": user['username']})
    async for record in cursor:
        if fs:
            if record.get("original_file_id"):
                try:
                    await fs.delete(ObjectId(record["original_file_id"]))
                except Exception: pass
            if record.get("heatmap_file_id"):
                try:
                    await fs.delete(ObjectId(record["heatmap_file_id"]))
                except Exception: pass
    
    # 2. Delete history and user
    await history_collection.delete_many({"username": user['username']})
    await db_client.brain_tumor_db.users.delete_one({"username": user['username']})
    
    return {"message": "Account deleted successfully"}

@app.get("/admin/stats")
async def get_admin_stats(admin: dict = Depends(get_current_admin_user)):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    # 1. Total Users
    total_users = await db_client.brain_tumor_db.users.count_documents({})
    
    # 2. Scans per User
    # This pipeline starts from the users collection to include all users,
    # then performs a "left join" to the history collection to count scans.
    pipeline = [
        {
            "$lookup": {
                "from": "history",
                "localField": "username",
                "foreignField": "username",
                "as": "user_scans"
            }
        },
        {
            "$project": {
                "_id": 0,
                "username": "$username",
                "scan_count": {"$size": "$user_scans"}
            }
        },
        {"$sort": {"scan_count": -1}}
    ]
    cursor = db_client.brain_tumor_db.users.aggregate(pipeline)
    scans_per_user = await cursor.to_list(length=None)
    
    return {"total_users": total_users, "user_activity": scans_per_user}

@app.get("/admin/users")
async def get_all_users(
    page: int = 1, 
    limit: int = 10, 
    search: str = Query(default=""),
    admin: dict = Depends(get_current_admin_user)
):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    query = {}
    if search:
        query["username"] = {"$regex": search, "$options": "i"}
    
    skip = (page - 1) * limit
    total_count = await db_client.brain_tumor_db.users.count_documents(query)
    
    cursor = db_client.brain_tumor_db.users.find(query, {"_id": 0, "username": 1, "is_admin": 1}).skip(skip).limit(limit)
    users = await cursor.to_list(length=limit)
    
    # Normalize is_admin for display
    for u in users:
        u['is_admin'] = str(u.get('is_admin', False)).lower() == 'true'
    return {"data": users, "total": total_count}

@app.put("/admin/users/{username}/role")
async def update_user_role(username: str, role_data: UserRoleUpdate, admin: dict = Depends(get_current_admin_user)):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    if username == admin['username'] and not role_data.is_admin:
         raise HTTPException(status_code=400, detail="Cannot remove your own admin privileges.")

    result = await db_client.brain_tumor_db.users.update_one(
        {"username": username},
        {"$set": {"is_admin": role_data.is_admin}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": f"User '{username}' role updated."}

@app.delete("/admin/users/{username}")
async def delete_user_by_admin(username: str, admin: dict = Depends(get_current_admin_user)):
    if db_client is None:
        raise HTTPException(status_code=503, detail="Database not connected")
        
    if username == admin['username']:
        raise HTTPException(status_code=400, detail="Cannot delete your own account from here.")

    # 1. Find all history to delete files (Copy logic from delete-account)
    cursor = history_collection.find({"username": username})
    async for record in cursor:
        if fs:
            if record.get("original_file_id"):
                try:
                    await fs.delete(ObjectId(record["original_file_id"]))
                except Exception: pass
            if record.get("heatmap_file_id"):
                try:
                    await fs.delete(ObjectId(record["heatmap_file_id"]))
                except Exception: pass
    
    # 2. Delete history and user
    await history_collection.delete_many({"username": username})
    result = await db_client.brain_tumor_db.users.delete_one({"username": username})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
        
    return {"message": f"User '{username}' deleted successfully"}

def process_dicom(file_bytes):
    """Reads a DICOM file buffer and converts it to a standard RGB numpy array."""
    try:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        img = ds.pixel_array
        
        # Normalize 16-bit/12-bit DICOM to 8-bit (0-255)
        if img.max() > 0:
            img = (img / img.max()) * 255.0
        img = np.uint8(img)
        
        # Convert to RGB if it's grayscale (most MRI scans are)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid DICOM file: {e}")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for model interpretability."""
    # Create a model that maps the input image to the activations of the last conv layer
    # and the output predictions
    try:
        # Attempt 1: Direct construction (Fastest, but fails on some loaded Sequential models)
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        print(f"DEBUG: Direct grad_model creation failed: {e}. Retrying with graph reconstruction.")
        # Attempt 2: Reconstruct graph (Robust for Sequential/Nested models)
        # We create a new input and re-pipe the data through the existing layers.
        # Since we reuse the layer instances, weights are preserved.
        input_shape = img_array.shape[1:] # (224, 224, 3)
        new_input = tf.keras.Input(shape=input_shape)
        x = new_input
        target_output = None
        
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                target_output = x
        
        grad_model = tf.keras.models.Model(new_input, [target_output, x])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron with regard to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap = heatmap / max_val
    return heatmap.numpy()

def is_valid_mri_scan(image: np.ndarray, min_brain_area_ratio=0.25) -> bool:
    """
    Performs heuristic checks to validate if an image is likely a brain MRI.
    1. Checks if the image is mostly grayscale.
    2. Checks if there is a significant central object (the brain).
    """
    # 1. Grayscale Check: A true color image will have a higher std dev across channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Using a sample of pixels for performance
        sample = image[::4, ::4, :]
        # Calculate the standard deviation of each pixel's channel values, then the std of those stds
        channel_diff_std = np.std(sample.std(axis=2))
        # This threshold is empirical. Grayscale images have a very low std dev between channels.
        if channel_diff_std > 15:
            print(f"Validation failed: Image appears to be a color photograph (channel std dev: {channel_diff_std:.2f})")
            return False

    # 2. Central Object Check
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Use Otsu's thresholding to automatically find a good threshold value
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Validation failed: No objects found in the image.")
        return False

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    total_image_area = image.shape[0] * image.shape[1]
    area_ratio = contour_area / total_image_area

    if area_ratio < min_brain_area_ratio:
        print(f"Validation failed: Largest object is too small (ratio: {area_ratio:.2f})")
        return False

    print("Validation passed: Image appears to be a valid MRI scan.")
    return True

@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    model_name: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    # For serverless: Lazily load model if it's not in memory (handles cold starts)
    if model_name not in MODELS:
        await load_single_model(model_name)

    selected_model = MODELS.get(model_name)
    config = MODEL_CONFIG.get(model_name)

    if not selected_model or not config:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model name '{model_name}'. Available models: {list(MODELS.keys())}"
        )

    contents = await file.read()
    
    # Store original file in GridFS
    original_file_id = None
    if fs:
        original_file_id = await fs.upload_from_stream(
            file.filename,
            io.BytesIO(contents),
            metadata={"content_type": file.content_type}
        )
    
    # 1. Load and Decode Image
    if file.filename.lower().endswith('.dcm'):
        image = process_dicom(contents)
    else:
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB

    # NEW: Validate if the image is a brain scan
    if not is_valid_mri_scan(image):
        # 422 Unprocessable Entity is a good status code for a validation error
        raise HTTPException(
            status_code=422, 
            detail="Invalid Image: The uploaded file does not appear to be a brain MRI scan. Please upload a valid scan."
        )

    # 2. Preprocess (Letterbox Resize)
    # Using the robust resizing from your data_utils.py
    processed_img = letterbox(image, new_shape=config['target_size'])
    
    # Prepare batch: (1, 224, 224, 3)
    img_batch = np.expand_dims(processed_img, axis=0)
    
    # Apply model-specific preprocessing (Mean subtraction/Scaling)
    # This converts RGB -> BGR and zero-centers it based on the model's training
    img_batch = config['preprocess_func'](img_batch.copy())

    # 3. Inference
    predictions = selected_model.predict(img_batch)
    confidence = float(predictions[0][0]) # Assuming Sigmoid output (0 to 1)
    
    # 4. Traffic Light Logic
    if confidence > HIGH_CONFIDENCE_THRESHOLD:
        label = "YES (Tumor Detected)"
        risk_level = "High" # Red
    elif confidence > LOW_CONFIDENCE_THRESHOLD:
        label = "Inconclusive / Needs Review"
        risk_level = "Medium" # Yellow
    else:
        label = "NO (Healthy)"
        risk_level = "Low" # Green

    # 5. Generate Grad-CAM (Best Effort)
    heatmap_b64 = None
    heatmap_file_id = None
    try:
        # Attempt to auto-detect the last convolutional layer
        last_conv_layer = None
        for layer in reversed(selected_model.layers):
            # 1. Check for direct Conv2D layer
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
            # 2. Check for a nested model/layer (like VGG16 base) that outputs 4D feature maps
            # This handles cases where the base model is wrapped (e.g. in Sequential API)
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if isinstance(shape, list): shape = shape[0]
                if len(shape) == 4: # (Batch, Height, Width, Channels)
                    last_conv_layer = layer.name
                    break
        
        if last_conv_layer:
            print(f"DEBUG: Found last convolutional layer: {last_conv_layer}")
            heatmap = make_gradcam_heatmap(img_batch, selected_model, last_conv_layer)
            
            if heatmap is not None:
                # Resize heatmap to match the original processed image size
                heatmap = cv2.resize(heatmap, (processed_img.shape[1], processed_img.shape[0]))
                
                # Convert to RGB heatmap (Jet colormap)
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) #COLORMAP_JET: A standard scheme where Blue represents low importance (the model ignored this area) and Red represents high importance (this area strongly influenced the "Tumor" prediction).
                
                # Superimpose heatmap on original image
                superimposed_img = heatmap * 0.4 + processed_img
                superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
                
                # Encode to Base64 for transport
                _, buffer = cv2.imencode('.jpg', superimposed_img)
                heatmap_bytes = buffer.tobytes()
                heatmap_b64 = base64.b64encode(heatmap_bytes).decode('utf-8')
                
                # Store heatmap in GridFS
                if fs:
                    heatmap_file_id = await fs.upload_from_stream(
                        f"heatmap_{file.filename}",
                        io.BytesIO(heatmap_bytes),
                        metadata={"content_type": "image/jpeg"}
                    )
            else:
                print("DEBUG: Grad-CAM heatmap is None (gradients failed).")
        else:
            print("DEBUG: No suitable 4D layer found for Grad-CAM.")
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")

    # Construct result dictionary
    result = {
        "filename": file.filename,
        "username": current_user.username,
        "label": label,
        "confidence": confidence,
        "risk_level": risk_level,
        "original_file_id": str(original_file_id) if original_file_id else None,
        "heatmap_file_id": str(heatmap_file_id) if heatmap_file_id else None,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat()
    }

    # Save to MongoDB (Fire and forget)
    if history_collection is not None:
        # We insert a copy to avoid modifying the return object if the driver adds an _id field
        await history_collection.insert_one(result.copy())

    result["heatmap_base64"] = heatmap_b64

    return result

@app.get("/history")
async def get_history(
    page: int = 1, 
    limit: int = 10, 
    search: str = Query(default=""),
    current_user: User = Depends(get_current_user)
):
    """Fetches analysis records from MongoDB with pagination and search."""
    if history_collection is None:
        return {"data": [], "total": 0}
    
    # Build query
    query = {"username": current_user.username}
    if search:
        search = search.strip()
        if search:
            query["filename"] = {"$regex": search, "$options": "i"}

    # Calculate how many documents to skip
    skip = (page - 1) * limit
    
    # Get total count for pagination UI
    total_count = await history_collection.count_documents(query)

    # Fetch records, apply sort, skip and limit
    # Exclude 'heatmap_base64' to keep the response light. We need '_id' for deletion.
    cursor = history_collection.find(query, {"heatmap_base64": 0}).sort("timestamp", -1).skip(skip).limit(limit)
    history = await cursor.to_list(length=limit)
    
    # Convert _id (ObjectId) to string so it can be sent as JSON
    for doc in history:
        doc['id'] = str(doc['_id'])
        del doc['_id']
        
    return {"data": history, "total": total_count}

@app.get("/files/{file_id}")
async def get_file(file_id: str, current_user: User = Depends(get_current_user)):
    """Retrieves a file from GridFS by its ID and streams it to the client."""
    if fs is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        oid = ObjectId(file_id)
        try:
            grid_out = await fs.open_download_stream(oid)
        except Exception:
             raise HTTPException(status_code=404, detail="File not found in GridFS")
             
        async def iterfile():
            while True:
                chunk = await grid_out.readchunk()
                if not chunk:
                    break
                yield chunk
                
        return StreamingResponse(
            iterfile(), 
            media_type=grid_out.metadata.get("content_type", "application/octet-stream")
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error retrieving file: {e}")

@app.delete("/history/{record_id}")
async def delete_history_record(record_id: str, current_user: User = Depends(get_current_user)):
    if history_collection is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Find record to get GridFS IDs
        record = await history_collection.find_one({"_id": ObjectId(record_id), "username": current_user.username})
        if not record:
            raise HTTPException(status_code=404, detail="Record not found or access denied")

        if fs:
            if record.get("original_file_id"):
                try:
                    await fs.delete(ObjectId(record["original_file_id"]))
                except Exception: pass
            if record.get("heatmap_file_id"):
                try:
                    await fs.delete(ObjectId(record["heatmap_file_id"]))
                except Exception: pass
        
        result = await history_collection.delete_one({"_id": ObjectId(record_id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        return {"message": "Record deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ID format or error: {e}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)