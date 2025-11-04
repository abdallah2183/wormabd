@echo off
echo ===========================================
echo WormGPT AI - Automatic Setup
echo ===========================================
echo.
echo This will set up WormGPT with GPU support.
echo Please run this as Administrator!
echo.
pause

:: Check if running as administrator
net session >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Running with administrator privileges...
) else (
    echo ERROR: Please run this script as Administrator!
    echo Right-click on this file and select "Run as administrator"
    pause
    exit /b 1
)

echo.
echo Step 1: Creating Python environment...
call conda create -n wormgpt-gpu python=3.10 -y
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo Step 2: Installing PyTorch with CUDA support...
call conda activate wormgpt-gpu
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install PyTorch
    pause
    exit /b 1
)

echo.
echo Step 3: Installing required packages...
pip install flask python-dotenv requests diffusers transformers accelerate xformers
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Step 4: Setting up environment...
if not exist "model_files" mkdir model_files
echo USE_CUDA=True>.env
echo AI_SERVER_API_KEY=My_Website_Secure_Key_123456>>.env
echo TEXT_MODEL=dolphin-2.9-llama3-8b-q8_0.gguf>>.env
echo IMAGE_MODEL=Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors>>.env
echo MODEL_PATH=model_files>>.env

echo.
echo ===========================================
echo Setup completed successfully!
echo ===========================================
echo.
echo To start WormGPT, run "start_server.bat"
echo.
pause
