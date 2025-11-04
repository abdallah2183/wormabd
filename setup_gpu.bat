@echo off
setlocal enabledelayedexpansion

echo === WormGPT GPU Setup ===
echo.

echo Step 1: Creating Python environment...
call conda create -n wormgpt-gpu python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create conda environment
    pause
    exit /b 1
)

echo Step 2: Activating environment...
call conda activate wormgpt-gpu
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)

echo Step 3: Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install PyTorch
    pause
    exit /b 1
)

echo Step 4: Installing requirements...
pip install flask python-dotenv requests diffusers transformers accelerate xformers
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo Step 5: Updating .env file...
echo USE_CUDA=True>>.env
echo AI_SERVER_API_KEY=My_Website_Secure_Key_123456>>.env
echo TEXT_MODEL=dolphin-2.9-llama3-8b-q8_0.gguf>>.env
echo IMAGE_MODEL=Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors>>.env
echo MODEL_PATH=model_files>>.env

if not exist "model_files" (
    mkdir model_files
    echo Created model_files directory
)

echo.
echo === Setup Completed Successfully! ===
echo.
echo To start the server, run:
echo    conda activate wormgpt-gpu
echo    python app.py

pause
