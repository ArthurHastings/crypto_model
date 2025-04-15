from imports import *


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("PROIECT_CRYPTOv1.2")

df = pd.read_csv(r"/Users/horialitan/Desktop/PythonAIRemote/Modul3/Course 2/Software/project/data_final.csv")

# df.to_csv("cleaned_dataset.csv", index=False)

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

X = df.cleaned_text
y = df.Sentiment
y = y.map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=68, test_size=0.2)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

y_train = np.array(y_train)


max_length = int(np.percentile([len(seq) for seq in X_train_seq], 95))
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    if word in glove_model:
        embedding_matrix[i] = glove_model[word]

embedding_layer = keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1,
                                         output_dim=100,
                                         weights=[embedding_matrix],
                                         input_length=max_length,
                                         trainable=True)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)

class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(class_weights_dict)

# optimizers = ["adam"]
# batch_sizes = [16, 32]
# epochs_list = [25, 30]
# neurons = [32, 64, 128]
# dropouts = [0.2, 0.4]
# learning_rates = [0.001, 0.0005, 0.0001]

optimizers = ["adamw"]
batch_sizes = [16]
epochs_list = [35]
neurons = [64]
dropouts = [0.3]
learning_rates = [0.0005]


param_combinations = list(itertools.product(optimizers, batch_sizes, epochs_list, neurons, dropouts, learning_rates))

for run_id, (optimizer, batch_size, epochs, neuron, dropout, learning_rate) in enumerate(param_combinations):


    with mlflow.start_run():
        
        run_name = f"Run_{run_id+1}_Opt-{optimizer}_BS-{batch_size}_Ep-{epochs}_Neurons-{neuron}_Dropout-{dropout}_LR-{learning_rate}"

        mlflow.set_tag("mlflow.runName", run_name)
        print(f"Starting {run_name}")

        model = keras.models.Sequential([
            embedding_layer,
            keras.layers.Bidirectional(keras.layers.LSTM(neuron, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.LSTM(neuron * 2, return_sequences=True)),
            keras.layers.Bidirectional(keras.layers.GRU(neuron * 2)),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(neuron * 2, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(3, activation="softmax")
        ])

        if optimizer == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optimizer == "adamw":
            optimizer = tf._optimizers.AdamW(learning_rate = learning_rate)

        model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=optimizer,
                    metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=13, restore_best_weights=True)
        # lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("neurons", neuron)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning rate", learning_rate)

        start_time = time.time()

        history = model.fit(X_train_pad, y_train, # [:len(X_train_pad)//10]
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test_pad, y_test),
                        class_weight=class_weights_dict,
                        callbacks=[early_stopping])

        
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)

        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
        # print(f"Accuracy: {accuracy:.6f}")
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

        mlflow.tensorflow.log_model(model, f"cnn_model_run{run_id+1}")

        mlflow.end_run()

model.save("sentiment_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

sentence_example = "Bitcoin price will increase tomorrow by 5%."
sentence_example_clean = preprocess_text(sentence_example)
sentence_example_seq = tokenizer.texts_to_sequences([sentence_example_clean])

sentence_example_pad = pad_sequences(sentence_example_seq, maxlen=max_length, padding='post')

pred = model.predict(sentence_example_pad)

print(pred)

predicted_label = np.argmax(pred)

label_reverse_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Predicted Sentiment: {label_reverse_mapping[predicted_label]}")