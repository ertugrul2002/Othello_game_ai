import numpy as np
import glob
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping





class GameNetworkSimple:
    def __init__(self, action_size=64, learning_rate=0.01):
        self.action_size = action_size
        self.model = self._build_model(learning_rate)

    def _build_model(self, learning_rate):
        # المدخلات: 3 طبقات (قطع اللاعب، قطع الخصم، دور اللعب) بمقاس 8x8
        # ملاحظة: تأكد من تنسيق البيانات (Channels Last) ليناسب TensorFlow
        # inputs = layers.Input(shape=(3, 8, 8)) 
        inputs = layers.Input(shape=(8, 8, 3))

        # (CNN Layers)
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, kernel_size=3, padding='same', activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, kernel_size=3, padding='same', activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # Policy Head: 
        policy_flat = layers.Flatten()(x)
        policy = layers.Dense(self.action_size, activation="softmax", name="policy")(policy_flat)

        # Value Head: 
        value_conv = layers.Conv2D(1, kernel_size=1, activation="relu")(x)
        value_flat = layers.Flatten()(value_conv)
        value_dense = layers.Dense(64, activation="relu")(value_flat)
        value = layers.Dense(1, activation="tanh", name="value")(value_dense)

        model = models.Model(inputs=inputs, outputs=[policy, value])
        model.compile(
            optimizer=optimizers.Adam(learning_rate),
            loss={"policy": "categorical_crossentropy", "value": "mean_squared_error"}
        )
        return model

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
            factor=0.2,       # تقليل الـ LR بنسبة 80% (يعني بيصير 0.002 لو كان 0.01)
            patience=2,       # استنى دورين لو ما في تحسن
            min_lr=0.0001,    # أقل قيمة مسموحة للـ LR
            verbose=1
        )
    
        # التوقف المبكر لمنع الـ Overfitting (الحفظ الصم)
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=4,       # إذا مر 4 أدوار بدون تحسن حقيقي، وقف التدريب
            restore_best_weights=True, # ارجع لأفضل نسخة من الأوزان مش آخر نسخة مخبصة
            verbose=1
        )

        # عملية التدريب
        history = self.model.fit(
            states, 
            {"policy": policies, "value": values},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,  # يقتطع 10% للتأكد من أن الموديل لا يحفظ فقط (Overfitting)
            shuffle=True           # ضروري جداً لخلط الحركات من ألعاب مختلفة
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
