from imports import *
from main_model import stock_symbol, url_api, tokenizer  # AAPL
from final_dataset_combine import period

mlflow.set_tracking_uri("http://localhost:5003")
mlflow.set_experiment("PROIECT_CRYPTO_PRICEv5.3.1")

data = pd.read_csv(f"apple_price_sentiment_{period}d.csv")
data = data[:-1]

X = data[['Open', 'High', 'Low', 'Volume', 'Sentiment headline', 'Sentiment summary']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input for LSTM (samples, timesteps, features)
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

optimizers = ["adam", "adamw"]
batch_sizes = [16, 32]
epochs_list = [30]
neurons = [32, 64, 128]
dropouts = [0, 0.3, 0.4]
learning_rates = [0.001, 0.0005, 0.0001]

param_combinations = list(itertools.product(optimizers, batch_sizes, epochs_list, neurons, dropouts, learning_rates))

for run_id, (optimizer, batch_size, epochs, neuron, dropout, learning_rate) in enumerate(param_combinations):
    with mlflow.start_run():
        run_name = f"Run_{run_id+1}_Opt-{optimizer}_BS-{batch_size}_Ep-{epochs}_Neurons-{neuron}_Dropout-{dropout}_LR-{learning_rate}"
        mlflow.set_tag("mlflow.runName", run_name)
        print(f"Starting {run_name}")

        model = keras.models.Sequential([
            keras.layers.LSTM(neuron, return_sequences=True, input_shape=(1, X_train_scaled.shape[2])),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(neuron // 2, return_sequences=False),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        if optimizer == "adam":
            optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "adamw":
            optimizer = tf.optimizers.AdamW(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=18, restore_best_weights=True)

        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("neurons", neuron)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning rate", learning_rate)

        start_time = time.time()

        history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)

        loss, accuracy = model.evaluate(X_test_scaled, y_test)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

        mlflow.tensorflow.log_model(model, f"lstm_model_run{run_id+1}")
        mlflow.end_run()


