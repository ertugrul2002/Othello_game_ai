import numpy as np
import matplotlib.pyplot as plt
import os
from othello_network_improved import GameNetworkImproved

def load_dataset(file_path):
    """
    تحميل البيانات من ملف NPZ
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        return None, None, None
    
    print(f"📂 Loading dataset from {file_path}...")
    data = np.load(file_path)
    
    states = data['states']
    policies = data['policies']
    values = data['values']
    
    print(f"✓ Loaded {len(states):,} samples")
    
    return states, policies, values


def plot_training_history(history, save_path="training_history_improved.png"):
    """
    رسم منحنيات التدريب
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Policy Loss
    axes[0, 0].plot(history.history['policy_loss'], label='Train Policy Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_policy_loss'], label='Val Policy Loss', linewidth=2)
    axes[0, 0].set_title('Policy Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value Loss
    axes[0, 1].plot(history.history['value_loss'], label='Train Value Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_value_loss'], label='Val Value Loss', linewidth=2)
    axes[0, 1].set_title('Value Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Policy Accuracy
    axes[1, 0].plot(history.history['policy_accuracy'], label='Train Policy Accuracy', linewidth=2)
    axes[1, 0].plot(history.history['val_policy_accuracy'], label='Val Policy Accuracy', linewidth=2)
    axes[1, 0].set_title('Policy Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Value MAE
    axes[1, 1].plot(history.history['value_mae'], label='Train Value MAE', linewidth=2)
    axes[1, 1].plot(history.history['val_value_mae'], label='Val Value MAE', linewidth=2)
    axes[1, 1].set_title('Value Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Training history plot saved to: {save_path}")
    plt.close()


def print_final_metrics(history):
    """
    طباعة الـ metrics النهائية
    """
    print(f"\n{'='*60}")
    print("FINAL TRAINING METRICS")
    print(f"{'='*60}")
    
    # آخر epoch
    last_epoch = len(history.history['loss']) - 1
    
    print(f"\n📊 Training Set (Epoch {last_epoch + 1}):")
    print(f"  - Policy Loss: {history.history['policy_loss'][-1]:.4f}")
    print(f"  - Policy Accuracy: {history.history['policy_accuracy'][-1]:.4f}")
    print(f"  - Value Loss: {history.history['value_loss'][-1]:.4f}")
    print(f"  - Value MAE: {history.history['value_mae'][-1]:.4f}")
    
    print(f"\n📊 Validation Set (Epoch {last_epoch + 1}):")
    print(f"  - Policy Loss: {history.history['val_policy_loss'][-1]:.4f}")
    print(f"  - Policy Accuracy: {history.history['val_policy_accuracy'][-1]:.4f}")
    print(f"  - Value Loss: {history.history['val_value_loss'][-1]:.4f}")
    print(f"  - Value MAE: {history.history['val_value_mae'][-1]:.4f}")
    
    # أفضل validation accuracy
    best_val_acc_idx = np.argmax(history.history['val_policy_accuracy'])
    best_val_acc = history.history['val_policy_accuracy'][best_val_acc_idx]
    
    print(f"\n🏆 Best Validation Policy Accuracy:")
    print(f"  - Epoch {best_val_acc_idx + 1}: {best_val_acc:.4f}")
    
    print(f"\n{'='*60}\n")


def main():
    """
    الدالة الرئيسية للتدريب
    """
    print(f"\n{'#'*60}")
    print("OTHELLO AI - IMPROVED TRAINING PIPELINE")
    print(f"{'#'*60}\n")
    
    # ========== 1. تحميل البيانات ==========
    data_file = "othello_full_1.5M.npz"
    states, policies, values = load_dataset(data_file)
    
    if states is None:
        print("❌ Failed to load dataset. Exiting...")
        return
    
    # ========== 2. إنشاء الشبكة المحسّنة ==========
    print(f"\n{'='*60}")
    print("Building improved CNN network...")
    print(f"{'='*60}\n")
    
    network = GameNetworkImproved(
        action_size=64, 
        learning_rate=0.0001  # تخفيض من 0.001 إلى 0.0001
    )
    
    # طباعة ملخص الموديل
    print("\n📋 Model Architecture:")
    network.get_model_summary()
    
    # ========== 3. التدريب ==========
    print(f"\n{'='*60}")
    print("Starting training process...")
    print(f"{'='*60}\n")
    
    history = network.train_model(
        states, 
        policies, 
        values,
        epochs=50,              # زيادة من 20 إلى 50
        batch_size=128,         # تقليل من 512 إلى 128
        use_augmentation=True   # تفعيل data augmentation
    )
    
    # ========== 4. حفظ الموديل ==========
    model_path = "othello_model_v4.keras"
    network.model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # ========== 5. رسم النتائج ==========
    plot_training_history(history)
    
    # ========== 6. طباعة الـ metrics النهائية ==========
    print_final_metrics(history)
    
    print(f"\n{'#'*60}")
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'#'*60}\n")
    
    print("📝 Next steps:")
    print("  1. Check the training history plot: training_history_improved.png")
    print("  2. Use the saved model: othello_model_v4.keras")
    print("  3. Test the model in actual games to evaluate performance")
    print()


if __name__ == "__main__":
    main()
