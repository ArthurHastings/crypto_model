from imports import *
from main_model import stock_symbol, url_api, tokenizer  # AAPL
from final_dataset_combine import period
import optuna
from main_model import stock_symbol, url_api, tokenizer
from final_dataset_combine import period
import shap

mlflow.set_tracking_uri("https://98c6-213-233-104-116.ngrok-free.app")
mlflow.set_experiment("PROIECT_CRYPTO_PRICEv8")


data = pd.read_csv(f"apple_price_sentiment_{period}d.csv")
data = data[:-1]
# for col in ['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']:
#     data[col] = data[col].shift(1)

# data = data.dropna().reset_index(drop=True)

X = data[['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

shap_sums = {name: 0 for name in ['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']}
num_trials = 5

def objective(trial):
    optimizer_name  = trial.suggest_categorical("optimizer",    ["adam", "adamw"])
    batch_size      = trial.suggest_categorical("batch_size",   [16, 32, 64])
    epochs          = trial.suggest_int(        "epochs",       10, 100, step=10)
    neurons         = trial.suggest_categorical("neurons",      [32, 64, 128])
    dropout_rate    = trial.suggest_float(      "dropout",      0.0, 0.5)
    learning_rate   = trial.suggest_loguniform( "learning_rate", 1e-10, 1e-2)

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
            keras.layers.LSTM(neurons, return_sequences=True, input_shape=(1, X_train.shape[2])),
            keras.layers.Dropout(dropout_rate),
            keras.layers.LSTM(neurons // 2, return_sequences=False),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        if optimizer_name == "adam":
            opt = tf.optimizers.Adam(learning_rate=learning_rate)
        else:
            opt = tf.optimizers.AdamW(learning_rate=learning_rate)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=18, restore_best_weights=True)

        start = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        # ------------------------------------------------------------------------------------------------------------------

        n_explain = min(100, X_test.shape[0])
        X_explain_seq = X_test[:n_explain]  # Shape: (n_explain, 1, 7)
        X_explain_flat = X_explain_seq.reshape((n_explain, X_explain_seq.shape[2]))  # Shape: (n_explain, 7)

        # 2. Define feature names manually
        feature_names = ['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']

        # 3. Wrap model prediction with reshaping
        def model_predict(data_flat):
            data_seq = data_flat.reshape((data_flat.shape[0], 1, data_flat.shape[1]))
            return model.predict(data_seq, verbose=0)

        # 4. Use background set of same shape (flattened 2D)
        background = X_train[:n_explain].reshape((n_explain, X_train.shape[2]))

        # 5. Explain
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_explain_flat)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # 6. Diagnostic: print mean absolute SHAP values per feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        print("📊 Mean absolute SHAP values:")
        for name, val in zip(feature_names, mean_shap):
            print(f"  {name:>8}: {float(val):.5f}")
        
        mlflow.log_metrics({
            "mean_shap_positive": float(mean_shap[feature_names.index("Positive")]),
            "mean_shap_negative": float(mean_shap[feature_names.index("Negative")]),
        })
        for trial_num in range(num_trials):
            for name, val in zip(feature_names, mean_shap):
                shap_sums[name] += val



        # ------------------------------------------------------------------------------------------------------------------
        mlflow.log_metric("training_time", time.time() - start)

        best_val_accuracy = max(history.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        
        model_path = f"lstm_model_trial_{trial.number}"
        mlflow.tensorflow.log_model(model, artifact_path=model_path)
        trial.set_user_attr("model_path", model_path)
        
        # Save correct run ID and path
        trial.set_user_attr("run_id", run.info.run_id)
        trial.set_user_attr("model_path", model_path)


        return best_val_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    print("✅ Best trial:")
    print(f"  Value (best_val_accuracy): {study.best_trial.value:.4f}")
    for key, val in study.best_trial.params.items():
        print(f"  {key}: {val}")

    avg_shap = {name: shap_sums[name] / num_trials for name in shap_sums}

    # Now, print the average SHAPs values after all trials
    print("\nAverage SHAP values after all trials:")
    for name, avg_val in avg_shap.items():
        print(f"  {name:>8}: {float(avg_val):.5f}")

    
    # ---------------------- PREDICTIONS ---------------------

    run_id = study.best_trial.user_attrs["run_id"]
    model_path = study.best_trial.user_attrs["model_path"]
    model_uri = f"runs:/{run_id}/{model_path}"

    best_model = mlflow.tensorflow.load_model(model_uri)

    data = pd.read_csv(f"apple_price_sentiment_{period}d.csv")

    latest_data = data.iloc[-1:]

    latest_features = latest_data[['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']]
    print("----------------- Todays data: -----------------")
    print(latest_features)
    latest_features_scaled = scaler.transform(latest_features)
    latest_features_scaled = latest_features_scaled.reshape((1, 1, latest_features_scaled.shape[1]))

    predicted_movement = best_model.predict(latest_features_scaled)
    predicted_class = "Up" if predicted_movement[0] > 0.5 else "Down"
    print(f"📈 Prediction for tomorrow: The price is predicted to go {predicted_class}.")

    with open("prediction_result.txt", "w") as f:
        f.write(f"📈 Prediction for tomorrow for {stock_symbol}: The price is predicted to go {predicted_class}.")

