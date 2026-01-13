# Setting Up ImageNet for exp_001 in Google Colab

This guide will help you get 2000 ImageNet validation images with seed=42 in Google Colab.

## ⚡ **Quick Setup (Recommended)**

Run these commands in your Colab notebook:

### **Step 1: Clone the Repository**
```python
!git clone https://github.com/YOUR_REPO/OAI-improved_SIMBA.git
%cd OAI-improved_SIMBA/GFCS
```

### **Step 2: Get Kaggle Credentials**
1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll to "API" section and click "Create New API Token"
3. This downloads `kaggle.json`
4. Upload it to Colab:

```python
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json here
```

### **Step 3: Run the Setup Script**
```python
!python setup_imagenet_colab.py
```

This will:
- ✅ Install Kaggle CLI
- ✅ Download ImageNet validation set (~6.3GB, ~10-15 minutes)
- ✅ Extract and organize images
- ✅ Create `./data/imagenet/val` directory

### **Step 4: Install Dependencies**
```python
!pip install torch torchvision numpy
```

### **Step 5: Run Your Experiment**
```python
!python run_experiment_from_config.py exp_001 --config_dir ./configs --device cuda
```

The code will automatically sample 2000 images using seed=42!

---

## 🚀 **Alternative: Faster Method Using torchvision**

If you don't want to download the full dataset, you can modify the config to use a smaller dataset:

### **Option A: Use CIFAR-10 for Testing**
```python
!python run_experiment_from_config.py exp_004 --config_dir ./configs --device cuda
```
This uses CIFAR-10 which downloads automatically (much smaller, ~170MB).

### **Option B: Create Minimal Test Config**

Create a test config that uses only 100 images to verify everything works:

```python
import json

# Load original config
with open('./configs/exp_001_baseline_resnet50_imagenet.json', 'r') as f:
    config = json.load(f)

# Modify for quick test
config['experiment_id'] = 'exp_001_test'
config['dataset']['num_images'] = 100  # Just 100 images for testing
config['attack']['max_queries'] = 1000  # Fewer queries for faster testing

# Save test config
with open('./configs/exp_001_test.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Created test config with 100 images")
```

Then run setup and experiment as normal.

---

## 📊 **What Happens During Sampling**

Your `run_experiment_from_config.py` already handles sampling correctly:

```python
# Line 184-206 in run_experiment_from_config.py
np.random.seed(seed)  # Sets seed to 42
indices = np.random.choice(len(dataset), size=min(num_images, len(dataset)), replace=False)
```

This ensures:
- ✅ Reproducible sampling with seed=42
- ✅ Exactly 2000 images (or fewer if dataset is smaller)
- ✅ No duplicates (replace=False)

---

## 🔧 **Troubleshooting**

### Problem: "Dataset path does not exist"
**Solution**: Make sure the setup script completed successfully and created `./data/imagenet/val`

### Problem: "Not enough images"
**Solution**: Reduce `num_images` in the config or download the full validation set

### Problem: "Kaggle API credentials not found"
**Solution**: Make sure you uploaded `kaggle.json` and ran the setup script

### Problem: "Out of memory in Colab"
**Solution**:
1. Use Colab Pro for more RAM
2. Reduce batch processing
3. Use fewer surrogate models
4. Test with fewer images first (e.g., 100)

---

## ⏱️ **Expected Time**

- Download ImageNet val: ~10-15 minutes (6.3GB)
- Extract and organize: ~5-10 minutes
- Run exp_001 (2000 images): ~2-4 hours depending on GPU

---

## 💡 **Pro Tips**

1. **Save to Google Drive**: Mount Drive and save results there so they persist
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Monitor Progress**: The script prints progress for each image

3. **Use GPU**: Make sure Colab is set to GPU runtime (Runtime → Change runtime type → GPU)

4. **Test First**: Run with 10-100 images first to verify everything works

---

## 🎯 **Final Command Sequence**

Here's the complete sequence to copy-paste into Colab:

```python
# 1. Clone repo
!git clone https://github.com/YOUR_REPO/OAI-improved_SIMBA.git
%cd OAI-improved_SIMBA/GFCS

# 2. Upload kaggle.json
from google.colab import files
uploaded = files.upload()

# 3. Setup ImageNet
!python setup_imagenet_colab.py

# 4. Install dependencies
!pip install torch torchvision numpy

# 5. Run experiment
!python run_experiment_from_config.py exp_001 --config_dir ./configs --device cuda
```

That's it! 🎉
