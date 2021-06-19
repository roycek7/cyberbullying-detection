import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

path = 'C:/Users/s4625266/PycharmProjects/coral/pickled_data'

input_size = (512, 512)
nb_samples = 1005
epochs = 500
batch_size = 32


def read(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


train_features, hi1_C, hi2_C, hi3_C, hi4_C, hi5_C, hi6_C = read(f'{path}/train_features.pkl'), \
                                                           read(f'{path}/hi1_C.pkl'), read(f'{path}/hi2_C.pkl'),\
                                                           read(f'{path}/hi3_C.pkl'), read(f'{path}/hi4_C.pkl'), \
                                                           read(f'{path}/hi5_C.pkl'), read(f'{path}/hi6_C.pkl')

train_features, test_features = train_test_split(train_features, shuffle=True, test_size=0.33, random_state=42)
hi1_C_train, hi1_C_test = train_test_split(hi1_C, shuffle=True, test_size=0.33, random_state=42)
hi2_C_train, hi2_C_test = train_test_split(hi2_C, shuffle=True, test_size=0.33, random_state=42)
hi3_C_train, hi3_C_test = train_test_split(hi3_C, shuffle=True, test_size=0.33, random_state=42)
hi4_C_train, hi4_C_test = train_test_split(hi4_C, shuffle=True, test_size=0.33, random_state=42)
hi5_C_train, hi5_C_test = train_test_split(hi5_C, shuffle=True, test_size=0.33, random_state=42)
hi6_C_train, hi6_C_test = train_test_split(hi6_C, shuffle=True, test_size=0.33, random_state=42)

img_input = layers.Input(shape=(256, 256, 2), name='input')

# --- block 1 ---
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# --- block 2 ---
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# --- coarse 1 branch ---
c_1_bch = layers.Flatten(name='h6_flatten')(x)
c_1_bch = layers.Dense(128, activation='relu', name='h6_fc_coral_1')(c_1_bch)
c_1_bch = layers.BatchNormalization()(c_1_bch)
c_1_bch = layers.Dropout(0.5)(c_1_bch)
c_1_bch = layers.Dense(128, activation='relu', name='h6_fc2')(c_1_bch)
c_1_bch = layers.BatchNormalization()(c_1_bch)
c_1_bch = layers.Dropout(0.5)(c_1_bch)
c_1_pred = layers.Dense(hi6_C_train.shape[1], activation='softmax', name='h6_predictions_coral')(c_1_bch)

# --- block 3 ---
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# --- coarse 2 branch ---
c_2_bch = layers.Flatten(name='h5_flatten')(x)
c_2_bch = layers.Dense(192, activation='relu', name='h5_fc_coral_1')(c_2_bch)
c_2_bch = layers.BatchNormalization()(c_2_bch)
c_2_bch = layers.Dropout(0.5)(c_2_bch)
c_2_bch = layers.Dense(192, activation='relu', name='h5_fc2')(c_2_bch)
c_2_bch = layers.BatchNormalization()(c_2_bch)
c_2_bch = layers.Dropout(0.5)(c_2_bch)
c_2_pred = layers.Dense(hi5_C_train.shape[1], activation='softmax', name='h5_predictions_coral')(c_2_bch)

# --- block 4 ---
x = layers.Conv2D(192, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(192, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# --- coarse 3 branch ---
c_3_bch = layers.Flatten(name='h4_flatten')(x)
c_3_bch = layers.Dense(256, activation='relu', name='h4_fc_coral_1')(c_3_bch)
c_3_bch = layers.BatchNormalization()(c_3_bch)
c_3_bch = layers.Dropout(0.5)(c_3_bch)
c_3_bch = layers.Dense(256, activation='relu', name='h4_fc2')(c_3_bch)
c_3_bch = layers.BatchNormalization()(c_3_bch)
c_3_bch = layers.Dropout(0.5)(c_3_bch)
c_3_pred = layers.Dense(hi4_C_train.shape[1], activation='softmax', name='h4_predictions_coral')(c_3_bch)

# --- block 5 ---
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# --- coarse 4 branch ---
c_4_bch = layers.Flatten(name='h3_flatten')(x)
c_4_bch = layers.Dense(384, activation='relu', name='h3_fc_coral_1')(c_4_bch)
c_4_bch = layers.BatchNormalization()(c_4_bch)
c_4_bch = layers.Dropout(0.5)(c_4_bch)
c_4_bch = layers.Dense(384, activation='relu', name='h3_fc2')(c_4_bch)
c_4_bch = layers.BatchNormalization()(c_4_bch)
c_4_bch = layers.Dropout(0.5)(c_4_bch)
c_4_pred = layers.Dense(hi3_C_train.shape[1], activation='softmax', name='h3_predictions_coral')(c_4_bch)

# --- block 6 ---
x = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(384, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

# --- coarse 5 branch ---
c_5_bch = layers.Flatten(name='h2_flatten')(x)
c_5_bch = layers.Dense(512, activation='relu', name='h2_fc_coral_1')(c_5_bch)
c_5_bch = layers.BatchNormalization()(c_5_bch)
c_5_bch = layers.Dropout(0.5)(c_5_bch)
c_5_bch = layers.Dense(512, activation='relu', name='h2_fc2')(c_5_bch)
c_5_bch = layers.BatchNormalization()(c_5_bch)
c_5_bch = layers.Dropout(0.5)(c_5_bch)
c_5_pred = layers.Dense(hi2_C_train.shape[1], activation='softmax', name='h2_predictions_coral')(c_5_bch)

# --- block 7 ---
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block7_conv3')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool')(x)

# --- fine block ---
c_6_bch = layers.Flatten(name='flatten')(x)
c_6_bch = layers.Dense(1024, activation='relu', name='fc_coral_1')(c_6_bch)
c_6_bch = layers.BatchNormalization()(c_6_bch)
c_6_bch = layers.Dropout(0.5)(c_6_bch)
c_6_bch = layers.Dense(1024, activation='relu', name='fc_coral_2')(c_6_bch)
c_6_bch = layers.BatchNormalization()(c_6_bch)
c_6_bch = layers.Dropout(0.5)(c_6_bch)
fine_pred = layers.Dense(hi1_C_train.shape[1], activation='softmax', name='predictions_coral')(c_6_bch)

# Training Model:
model = Model(img_input, [c_1_pred, c_2_pred, c_3_pred, c_4_pred, c_5_pred, fine_pred], name='hierarchy')
model.summary()

alpha = K.variable(value=0.166, dtype="float32", name="alpha")
beta = K.variable(value=0.166, dtype="float32", name="beta")
gamma = K.variable(value=0.166, dtype="float32", name="gamma")
delta = K.variable(value=0.166, dtype="float32", name="delta")
epsilon = K.variable(value=0.166, dtype="float32", name="epsilon")
zeta = K.variable(value=0.170, dtype="float32", name="zeta")

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              loss_weights=[alpha.numpy(), beta.numpy(), gamma.numpy(), delta.numpy(), epsilon.numpy(), zeta.numpy()],
              metrics=['accuracy'])

model.fit(train_features, [hi6_C_train, hi5_C_train, hi4_C_train, hi3_C_train, hi2_C_train, hi1_C_train],
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(test_features, [hi6_C_test, hi5_C_test, hi4_C_test, hi3_C_test, hi2_C_test, hi1_C_test]),
          callbacks=[EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)],
          verbose=2)

model.save('C:/Users/s4625266/PycharmProjects/coral/log/coral_classification.h5')

score = model.evaluate(test_features, [hi6_C_test, hi5_C_test, hi4_C_test, hi3_C_test, hi2_C_test, hi1_C_test],
                       verbose=0)
print('score is: ', score)
