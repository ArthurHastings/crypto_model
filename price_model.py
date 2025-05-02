from imports import *
from final_dataset_combine import period
import optuna

stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD", "BA", "JPM", "DIS", "V", "NKE"]
mlflow_ngrok = os.getenv("MLFLOW_NGROK", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_ngrok)
print(f"MLFLOW URL: {mlflow_ngrok}")

for stock_symbol in stock_list:

    mlflow.set_experiment(f"MODEL_{stock_symbol}v1")

    data = pd.read_csv(f"{stock_symbol}_price_sentiment.csv")
    data = data[:-1]

    X = data[['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    num_trials = 25

    def objective(trial):
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        epochs = trial.suggest_int("epochs", 100, 300, step=50)
        neurons = trial.suggest_categorical("neurons", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)

        with mlflow.start_run(nested=True) as run:
            mlflow.set_tag("mlflow.runName", f"Optuna_trial_{trial.number}")
            mlflow.log_params({
                "optimizer": optimizer_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "neurons": neurons,
                "dropout": dropout_rate,
                "learning_rate": learning_rate,
            })

            model = keras.models.Sequential([
                keras.layers.Input(shape=(1, X_train.shape[2])),
                keras.layers.Bidirectional(keras.layers.LSTM(neurons, return_sequences=True)),
                keras.layers.LayerNormalization(),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Bidirectional(keras.layers.LSTM(neurons // 2)),
                keras.layers.LayerNormalization(),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            opt = tf.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == "adam" else tf.optimizers.AdamW(learning_rate=learning_rate)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            start = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )

            mlflow.log_metric("training_time", time.time() - start)

            best_val_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric("best_val_accuracy", best_val_accuracy)

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)

            model_path = f"lstm_model_trial_{trial.number}"
            mlflow.tensorflow.log_model(model, artifact_path=model_path)

            trial.set_user_attr("model_path", model_path)
            trial.set_user_attr("run_id", run.info.run_id)

            return best_val_accuracy

    if __name__ == "__main__":
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=num_trials)

        print(f"✅ Best trial for {stock_symbol}:")
        print(f"  Value (best_val_accuracy): {study.best_trial.value:.4f}")
        for key, val in study.best_trial.params.items():
            print(f"  {key}: {val}")
