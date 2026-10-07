import copy
import glob
import math
import random
from typing import List, Tuple, Optional
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===== الأوزان الاستراتيجية للعبة Othello =====
WEIGHTS = np.array([
        [100, -20,  10,   5,   5,  10, -20, 100],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [ 10,  -2,   5,   1,   1,   5,  -2,  10],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [  5,  -2,   1,   0,   0,   1,  -2,   5],
        [ 10,  -2,   5,   1,   1,   5,  -2,  10],
        [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
        [100, -20,  10,   5,   5,  10, -20, 100]
    ])

# ===== Position Bias: تخلي الموديل يركز على الزوايا =====
# هذا بيساعد الموديل يتعلم أسرع إن الزوايا مهمة
POSITION_BIAS = WEIGHTS / WEIGHTS.max()  # تطبيع بين 0 و 1


# ===== الشبكة العصبية المحسّنة =====
class GameNetworkSimple:
    def __init__(self, action_size=64, learning_rate=0.001):
        self.action_size = action_size
        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate):
        """
        بناء الموديل مع التحسينات:
        1. Position Bias في Policy Head
        2. تحسين Value Head بمعلومات الموقع
        3. Weighted Loss للزوايا
        """
        inputs = layers.Input(shape=(8, 8, 3))

        # ===== CNN Layers =====
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, kernel_size=3, padding='same', activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # ===== Policy Head مع Position Bias =====
        policy_flat = layers.Flatten()(x)
        policy_dense = layers.Dense(256, activation="relu")(policy_flat)
        policy_dense = layers.Dropout(0.3)(policy_dense)
        policy_logits = layers.Dense(self.action_size, activation="linear")(policy_dense)
        
        # 🔥 إضافة Position Bias: تخلي الموديل يركز على الزوايا
        # نحول POSITION_BIAS لـ tensor ونضيفه للـ logits
        position_bias_flat = tf.constant(POSITION_BIAS.flatten(), dtype=tf.float32)
        policy_with_bias = layers.Lambda(
            lambda x: x + position_bias_flat * 20.0,  # 50% من الأهمية للـ bias
            name="policy_with_bias"
        )(policy_logits)
        
        policy = layers.Softmax(name="policy")(policy_with_bias)

        # ===== Value Head محسّن مع معلومات الموقع =====
        # الجزء الأول: معالجة عادية
        value_conv = layers.Conv2D(32, kernel_size=3, padding='same', activation="relu")(x)
        value_conv = layers.Conv2D(1, kernel_size=1, activation="relu")(value_conv)
        value_flat = layers.Flatten()(value_conv)
        value_dense = layers.Dense(64, activation="relu")(value_flat)
        
        # 🔥 الجزء الثاني: إضافة معلومات الموقع على اللوحة
        # هذا يخلي الموديل يعرف إن الزوايا مهمة
        position_features = layers.Conv2D(8, kernel_size=1, activation="relu")(x)
        position_features = layers.Flatten()(position_features)
        position_features = layers.Dense(32, activation="relu")(position_features)
        
        # دمج المعلومات
        combined = layers.Concatenate()([value_dense, position_features])
        combined = layers.Dense(64, activation="relu")(combined)
        combined = layers.Dropout(0.2)(combined)
        value = layers.Dense(1, activation="tanh", name="value")(combined)

        model = models.Model(inputs=inputs, outputs=[policy, value])
        
        # ===== Custom Loss مع وزن للزوايا =====
        # الموديل بيتعاقب أكثر إذا اختار حركة سيئة في الزاوية
        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss={
                "policy": "categorical_crossentropy",  # استخدام الـ loss العادي (بسيط وفعال)
                "value": "mean_squared_error"
            },
            loss_weights={"policy": 1.0, "value": 0.5}  # Value أقل أهمية من Policy
        )
        return model

    def _weighted_categorical_crossentropy(self):
        """
        دالة Loss مخصصة: تعطي وزن أكبر للزوايا
        الموديل بيتعاقب أكثر إذا اختار حركة سيئة في الزاوية
        """
        # إنشاء الأوزان مرة واحدة
        corner_weights = np.array([
            [10, 3, 3, 3, 3, 3, 3, 10],
            [3,  1, 1, 1, 1, 1, 1, 3],
            [3,  1, 1, 1, 1, 1, 1, 3],
            [3,  1, 1, 1, 1, 1, 1, 3],
            [3,  1, 1, 1, 1, 1, 1, 3],
            [3,  1, 1, 1, 1, 1, 1, 3],
            [3,  1, 1, 1, 1, 1, 1, 3],
            [10, 3, 3, 3, 3, 3, 3, 10]
        ]).flatten().astype(np.float32)
        
        corner_weights_tensor = tf.constant(corner_weights, dtype=tf.float32)
        
        def weighted_cce(y_true, y_pred):
            # حساب الـ loss العادي
            cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
            
            # إعادة تشكيل الأوزان لتطابق batch size
            # cce له shape [batch_size], نحتاج نضرب كل عنصر في الأوزان
            batch_size = tf.shape(y_pred)[0]
            
            # نحسب weighted loss لكل عنصر في الـ batch
            weighted_loss = 0.0
            for i in range(64):
                weighted_loss += cce[:, i] * corner_weights_tensor[i] if len(cce.shape) > 1 else cce * corner_weights_tensor[i]
            
            # طريقة أبسط وأسرع
            if len(y_pred.shape) > 1:
                # y_pred has shape [batch_size, 64]
                weighted = tf.reduce_sum(y_true * corner_weights_tensor * tf.math.log(y_pred + 1e-7), axis=-1)
                return -tf.reduce_mean(weighted)
            else:
                return cce
        
        return weighted_cce

    def predict(self, state):
        state = np.expand_dims(state, axis=0)
        policy, value = self.model.predict(state, verbose=0)
        return policy[0], value[0][0]
    

    def combine_cluster_results(self):
        files = glob.glob("cluster/data*.npz")
        all_s, all_p, all_v = [], [], []
        if not files:
            print("No data files found.")
            return None, None, None
        for f in files:
            data = np.load(f)
            all_s.append(data['states'])
            all_p.append(data['policies'])
            all_v.append(data['values'])
            
        # دمج الكل في ملف واحد عملاق للتدريب
        final_states = np.concatenate(all_s)
        final_policies = np.concatenate(all_p)
        final_values = np.concatenate(all_v)
        
        return final_states, final_policies, final_values

    def train_model(self, states, policies, values, epochs=20, batch_size=64):
        """
        تدريب الموديل على البيانات المجمعة
        """
        print(f"Starting training on {len(states)} samples...")
        
        # تحويل البيانات إلى تنسيق float32 لضمان استقرار التدريب
        states = states.astype('float32')
        policies = policies.astype('float32')
        values = values.astype('float32')
        if states.shape[1:] == (3, 8, 8):
            print("Converting states from (3,8,8) to (8,8,3)")
            states = np.transpose(states, (0, 2, 3, 1))

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,       # تقليل الـ LR بنسبة 80%
            patience=2,       # استنى دورين لو ما في تحسن
            min_lr=0.0001,    # أقل قيمة مسموحة للـ LR
            verbose=1
        )
    
        # التوقف المبكر لمنع الـ Overfitting
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=4,       # إذا مر 4 أدوار بدون تحسن، وقف التدريب
            restore_best_weights=True,
            verbose=1
        )

        # عملية التدريب
        history = self.model.fit(
            states, 
            {"policy": policies, "value": values},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=True,
            callbacks=[reduce_lr, early_stop]
        )
        return history

    def augment_data(self, states, policies, values):
        augmented_states = []
        augmented_policies = []
        augmented_values = []

        for s, p, v in zip(states, policies, values):
            # p بتوصل كـ vector طوله 64، لازم نحولها لمصفوفة 8x8 عشان ندورها
            p_board = p.reshape(8, 8)
            
            for i in range(4): # 4 تدويرات: 0, 90, 180, 270 درجة
                # 1. تدوير الحالة والسياسة
                s_rot = np.rot90(s, k=i, axes=(1, 2))
                p_rot = np.rot90(p_board, k=i)
                
                augmented_states.append(s_rot)
                augmented_policies.append(p_rot.flatten())
                augmented_values.append(v)

                # 2. قلب اللوحة (Flip) يمين-يسار لكل تدويرة
                s_flip = np.flip(s_rot, axis=2)
                p_flip = np.flip(p_rot, axis=1)
                
                augmented_states.append(s_flip)
                augmented_policies.append(p_flip.flatten())
                augmented_values.append(v)

        return np.array(augmented_states), np.array(augmented_policies), np.array(augmented_values)




# ===== Main Training Script =====
if __name__ == "__main__":
    game = Othello()
    game_net = GameNetworkImproved(action_size=64)

    # 2. تجميع البيانات من الملفات التي حفظتها
    states, policies, values = game_net.combine_cluster_results()

    if states is None or policies is None or values is None:
        print("No data available for training. Exiting.")
        exit()

    # 4. بدء التدريب
    if states is not None:
        print(f"Original data: {len(states)} samples")
        
        # 2. مضاعفة البيانات 8 مرات
        states, policies, values = game_net.augment_data(states, policies, values)
        print(f"Augmented data: {len(states)} samples")

        # 3. التدريب بالبيانات الكبيرة
        # 🔥 زيادة الـ epochs لأن الموديل محسّن الآن
        game_net.train_model(states, policies, values, epochs=30, batch_size=128)

    # 5. حفظ الموديل بعد التدريب
    game_net.model.save('my_othello_model_improved_v1.keras')
    print("✓ Model saved successfully as 'my_othello_model_improved_v1.keras'")
