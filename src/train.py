# train.py

from src.dataset import train_images, test_images, train_ages, test_ages, train_genders, test_genders
from src.model import get_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Assuming all images are of the same size
input_shape = train_images.shape[1:]  # This should match the shape of your preprocessed images

# Instantiate the model
model = get_model(input_shape)

# Path to save the best model
best_model_path = 'models/age_gender_model.h5'

# Callbacks
checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

# Train the model
history = model.fit(
    train_images,
    {'age_output': train_ages, 'gender_output': train_genders},
    validation_data=(test_images, {'age_output': test_ages, 'gender_output': test_genders}),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Optional: Save the final model
# model.save('models/final_age_gender_model.h5')

print("Training complete.")

# Optionally, add code here to plot training history, analyze results, etc.
