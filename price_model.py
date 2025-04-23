from imports import *
from main_model import stock_symbol, url_api, tokenizer  # AAPL
from final_dataset_combine import period
import optuna
from main_model import stock_symbol, url_api, tokenizer
from final_dataset_combine import period
import shap

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("PROIECT_CRYPTO_PRICEv7")


data = pd.read_csv(f"apple_price_sentiment_{period}d.csv")
data = data[:-1]
X = data[['Open', 'High', 'Low', 'Volume', 'Negative', "Neutral", "Positive"]]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

def objective(trial):
    optimizer_name  = trial.suggest_categorical("optimizer",    ["adam", "adamw"])
    batch_size      = trial.suggest_categorical("batch_size",   [16, 32, 64])
    epochs          = trial.suggest_int(        "epochs",       10, 100, step=10)
    neurons         = trial.suggest_categorical("neurons",      [32, 64, 128])
    dropout_rate    = trial.suggest_float(      "dropout",      0.0, 0.5)
    learning_rate   = trial.suggest_loguniform( "learning_rate", 1e-10, 1e-2)

    with mlflow.start_run(nested=True):
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

        X_full = data[['Open', 'High', 'Low', 'Volume', 'Negative', "Neutral", "Positive"]]

        # Match X_test to X_full for feature names
        n_explain = min(100, X_test.shape[0])
        X_explain_seq = X_test[:n_explain]
        X_explain_flat = X_explain_seq.reshape((n_explain, X_explain_seq.shape[2]))

        # Get feature names from the original DataFrame
        feature_names = X_full.columns.tolist()

        # KernelExplainer needs 2D input, and model output must be 1D or 2D
        def model_predict(data_flat):
            data_seq = data_flat.reshape((data_flat.shape[0], 1, data_flat.shape[1]))
            return model.predict(data_seq, verbose=0)

        # Use a representative background sample (also 2D flat input)
        background = X_train[:n_explain].reshape((n_explain, X_train.shape[2]))

        # SHAP explanation
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_explain_flat)

        # Handle classification output format
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # use the first output class

        # Plot and save the SHAP summary
        if shap_values is not None and shap_values.shape[1] == len(feature_names):
            shap.summary_plot(shap_values, X_explain_flat, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig("shap_summary_plot_fixed.png")
            plt.close()
            print("✅ SHAP summary plot saved as 'shap_summary_plot_fixed.png'")
        else:
            print(f"⚠️ SHAP values have unexpected shape: {np.shape(shap_values)}")

        # ------------------------------------------------------------------------------------------------------------------
        mlflow.log_metric("training_time", time.time() - start)

        best_val_accuracy = max(history.history['val_accuracy'])
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        
        mlflow.tensorflow.log_model(model, artifact_path=f"lstm_model_trial_{trial.number}")

        return best_val_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("✅ Best trial:")
    print(f"  Value (best_val_accuracy): {study.best_trial.value:.4f}")
    for key, val in study.best_trial.params.items():
        print(f"  {key}: {val}")