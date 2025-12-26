echo "[1/4] Creating venv..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
echo "[2/4] Installing pip dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create datasets directory
echo "[3/4] Creating datasets directory..."
mkdir -p datasets

# Download Kaggle dataset
echo "[4/4] Downloading Kaggle dataset..."
kaggle datasets download kamino/largescale-common-watermark-dataset -p datasets

# Unzip any zip files if present
cd datasets
if ls *.zip 1> /dev/null 2>&1; then
  for f in *.zip; do
    unzip -n "$f"
  done
fi
cd ..
echo "Installation complete!"