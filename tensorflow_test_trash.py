cat > test_tensorflow_cuda.py << 'EOF'
import tensorflow as tf
from tensorflow.keras import layers, models

# Check if CUDA (GPU) is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. Using CUDA!")
    print(f"GPU device: {tf.config.list_physical_devices('GPU')}")
else:
    print("CUDA is not available. Using CPU.")

# Create a simple neural network model
model = models.Sequential([
    layers.Dense(50, activation='relu', input_shape=(10,)),  # Input layer with 10 features
    layers.Dense(1)  # Output layer with 1 output
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Create dummy data
inputs = tf.random.normal([100, 10])  # 100 samples, 10 features each
targets = tf.random.normal([100, 1])  # 100 target values

# Train the model
model.fit(inputs, targets, epochs=10, verbose=1)

# Check if the GPU is being utilized
if tf.config.list_physical_devices('GPU'):
    print("CUDA is working with TensorFlow!")
else:
    print("CUDA is not available. Running on CPU.")
EOF
