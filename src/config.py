# config.py

# Dataset Information
DATASET_PATH = '/data/UTKFace'
TRAIN_TEST_SPLIT_RATIO = 0.2

# Image Information
IMAGE_WIDTH = 198
IMAGE_HEIGHT = 198
IMAGE_CHANNELS = 3  # assuming color images

# Training Information
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model Checkpoints and Save Paths
MODEL_CHECKPOINT_PATH = 'models/checkpoints'
BEST_MODEL_PATH = 'models/best_model.h5'
FINAL_MODEL_PATH = 'models/final_model.h5'

# Early Stopping
EARLY_STOPPING_PATIENCE = 5

# Feature Settings
AGE_CLASSES = ['0-25', '26-50', '51-75', '76+']  # Example if using categorical age ranges
NUM_GENDER_CLASSES = 2

# Add any additional configuration settings as needed
