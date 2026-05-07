import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("merged_climate_soil_recommended.csv")

# ==============================
# 2. DEFINE COLUMNS
# ==============================
climate_cols = ['T2M', 'RH', 'WS', 'SWGDN']
soil_cols = ['pH', 'EC (ds/m)', 'Ca²⁺ (ppm)', 'Mg²⁺ (ppm)', 'Na⁺ (ppm)', 
             'K⁺ (ppm)', 'CaCO₃ (%)', 'ESP', 'Water_TDS (ppm)']
target_col = 'crop recomended'

# ==============================
# 3. CLEAN TARGET
# ==============================
df = df.dropna(subset=[target_col]).reset_index(drop=True)
df[target_col] = df[target_col].fillna("")
df['target_list'] = df[target_col].apply(lambda x: x.split(', ') if x != "" else [])

# ==============================
# 4. ENCODE TARGET (Multi-label)
# ==============================
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['target_list'])

# ==============================
# 5. SCALE FEATURES
# ==============================
scaler_climate = StandardScaler()
scaler_soil = StandardScaler()

X_climate = scaler_climate.fit_transform(df[climate_cols])
X_soil = scaler_soil.fit_transform(df[soil_cols])

# ==============================
# 6. CREATE SEQUENCES (LSTM)
# ==============================
def create_sequences_by_group(df, X_climate, X_soil, y, time_steps=7):
    Xc, Xs, Y = [], [], []
    for auger_id, group in df.groupby('Auger'):
        indices = group.index
        for i in range(len(indices) - time_steps):
            idx_seq = indices[i:i+time_steps]
            idx_target = indices[i+time_steps]
            
            Xc.append(X_climate[idx_seq])
            Xs.append(X_soil[idx_target])
            Y.append(y[idx_target])
            
    return np.array(Xc), np.array(Xs), np.array(Y)

time_steps = 7
Xc, Xs, y_seq = create_sequences_by_group(df, X_climate, X_soil, y, time_steps)

# ==============================
# 7. TRAIN TEST SPLIT
# ==============================
Xc_train, Xc_test, Xs_train, Xs_test, y_train, y_test = train_test_split(
    Xc, Xs, y_seq, test_size=0.2, random_state=42
)

# ==============================
# 7.5 COMPUTE SAMPLE WEIGHTS (Alternative to SMOTE)
# ==============================
class_counts = np.sum(y_train, axis=0)
total_samples = y_train.shape[0]
num_classes = len(mlb.classes_)

class_weights = total_samples / (num_classes * (class_counts + 1))

sample_weights = []
for i in range(total_samples):
    active_classes = np.where(y_train[i] == 1)[0]
    if len(active_classes) > 0:
        w = np.max(class_weights[active_classes])
    else:
        w = 1.0
    sample_weights.append(w)

sample_weights = np.array(sample_weights)
print("\n--- Applied Dynamic Sample Weights to combat Imbalance ---")

# ==============================
# 8. BUILD MODEL (Strong Regularization to fix Overfitting)
# ==============================
climate_input = layers.Input(shape=(time_steps, len(climate_cols)), name="climate_input")
x1 = layers.LSTM(32, kernel_regularizer=l2(0.005), recurrent_regularizer=l2(0.005))(climate_input)
x1 = layers.Dropout(0.5)(x1)

soil_input = layers.Input(shape=(len(soil_cols),), name="soil_input")
x2 = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.005))(soil_input)
x2 = layers.Dropout(0.5)(x2)

x = layers.concatenate([x1, x2])
x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.005))(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(num_classes, activation='sigmoid', name="output")(x)

model = models.Model(inputs=[climate_input, soil_input], outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

model.summary()

# ==============================
# 9. TRAIN MODEL (with ReduceLROnPlateau)
# ==============================
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

history = model.fit(
    [Xc_train, Xs_train],
    y_train,
    sample_weight=sample_weights,
    validation_split=0.1,
    epochs=50,
    batch_size=64, # Larger batch size to stabilize gradients
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# ==============================
# 10. EVALUATION
# ==============================
loss, acc, prec, rec = model.evaluate([Xc_test, Xs_test], y_test, verbose=0)
print(f"\n====================================")
print(f"Test Accuracy:  {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall:    {rec:.4f}")
print(f"====================================")

# ==============================
# 11. SAVE MODEL & TOOLS
# ==============================
model.save("final version model lstm.keras")
joblib.dump(scaler_climate, "scaler_climate.save")
joblib.dump(scaler_soil, "scaler_soil.save")
joblib.dump(mlb, "mlb_dual.save")

# ==============================
# 12. DECISION SUPPORT SYSTEM FUNCTION
# ==============================
def decision_support_system(model, Xc_sample, Xs_sample, mlb, top_k=3):
    preds = model.predict([Xc_sample, Xs_sample], verbose=0)[0]
    
    print("\n--- Debug: All Crop Probabilities ---")
    for crop, score in zip(mlb.classes_, preds):
        print(f"  > {crop}: {score:.3f}")
    print("-------------------------------------")
    
    preds = np.clip(preds, 0.01, 0.95)
    
    top_indices = np.argsort(preds)[::-1][:top_k]
    recommendations = [(mlb.classes_[i], preds[i]) for i in top_indices]
            
    return recommendations

# ==============================
# 13. TEST RECOMMENDATION
# ==============================
sample_climate = Xc_test[0:1]
sample_soil = Xs_test[0:1]
recs = decision_support_system(model, sample_climate, sample_soil, mlb, top_k=3)

print("\n🌍 Decision Support System Output (Top 3):")
for idx, (crop, score) in enumerate(recs, 1):
    if score >= 0.70:
        status = "مناسب جداً (Highly Suitable)"
    elif score >= 0.40:
        status = "مناسب (Suitable)"
    else:
        status = "يتحمل الظروف (Tolerates Conditions)"
        
    print(f"{idx}. {crop} ({score:.2f}) → {status}")

# ==============================
# 14. PLOT TRAINING HISTORY
# ==============================
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Weighted Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=300)
print("\nTraining plots saved as 'training_history.png'")
