import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------
# Load Dataset
file_path = 'Input_Data.csv'
df = pd.read_csv(file_path)

# Define feature and target columns
y_features = [f'y_U{i}' for i in range(1, 9)] + [f'y_L{i}' for i in range(1, 9)]
condition_features = ['alpha', 'Mach', 'Cl', 'Cm']

# Extract data
X = df[y_features].values
conditions = df[condition_features].values

# Normalize data
shape_scaler = MinMaxScaler()
condition_scaler = StandardScaler()

X_normalized = shape_scaler.fit_transform(X)
conditions_normalized = condition_scaler.fit_transform(conditions)

# -------------------------------
# 2. Build Conditional VAE Model
# -------------------------------
LATENT_DIM = 16
INPUT_DIM = X_normalized.shape[1]
CONDITION_DIM = conditions_normalized.shape[1]

# Encoder
condition_input = tf.keras.layers.Input(shape=(CONDITION_DIM,), name='condition_input')
shape_input = tf.keras.layers.Input(shape=(INPUT_DIM,), name='shape_input')
combined_input = tf.keras.layers.Concatenate()([shape_input, condition_input])

x = tf.keras.layers.Dense(64, activation='relu')(combined_input)
x = tf.keras.layers.Dense(32, activation='relu')(x)

z_mean = tf.keras.layers.Dense(LATENT_DIM, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(LATENT_DIM, name='z_log_var')(x)

# Sampling Layer
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], LATENT_DIM))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling()([z_mean, z_log_var])

# Decoder
latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name='latent_input')
combined_decoder_input = tf.keras.layers.Concatenate()([latent_input, condition_input])

decoder_hidden = tf.keras.layers.Dense(32, activation='relu')(combined_decoder_input)
decoder_hidden = tf.keras.layers.Dense(64, activation='relu')(decoder_hidden)
decoder_output = tf.keras.layers.Dense(INPUT_DIM, activation='sigmoid')(decoder_hidden)

# Models
encoder = tf.keras.layers.Model([shape_input, condition_input], [z_mean, z_log_var, z], name='encoder')
decoder = tf.keras.layers.Model([latent_input, condition_input], decoder_output, name='decoder')

# VAE Model
shape_decoded = decoder([z, condition_input])
vae = tf.keras.layers.Model([shape_input, condition_input], shape_decoded, name='vae')

# VAE Loss
reconstruction_loss = tf.keras.losses.MeanSquaredError()(shape_input, shape_decoded)
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae_loss = reconstruction_loss + kl_loss

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# -------------------------------
# 3. Train the Model
# -------------------------------
history = vae.fit(
    [X_normalized, conditions_normalized],
    X_normalized,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# -------------------------------
# 4. Generate Airfoils
# -------------------------------
def generate_airfoil(condition_input):
    condition_input = condition_scaler.transform([condition_input])
    random_latent_vector = np.random.normal(size=(1, LATENT_DIM))
    generated_shape = decoder.predict([random_latent_vector, condition_input])
    return shape_scaler.inverse_transform(generated_shape)

# Example Usage
condition_example = [-2.0, 0.65, 0.1, 0.05]  # Example aerodynamic targets
generated_airfoil = generate_airfoil(condition_example)
print("Generated Airfoil Coordinates:")
print(generated_airfoil)

# Save Models
encoder.save('encoder_model.h5')
decoder.save('decoder_model.h5')
vae.save('vae_model.h5')

# -------------------------------
# 5. Visualization (Optional)
# -------------------------------
import matplotlib.pyplot as plt

def plot_airfoil(coords):
    x_coords = np.linspace(0, 1, 8)
    y_upper = coords[0, :8]
    y_lower = coords[0, 8:]
    plt.plot(x_coords, y_upper, label='Upper Surface')
    plt.plot(x_coords, y_lower, label='Lower Surface')
    plt.legend()
    plt.title('Generated Airfoil Shape')
    plt.show()

plot_airfoil(generated_airfoil)
