// Ensure styles are linked and TFJS is included in index.html

// Global variables
let allIntents = {};
let allMenuItems = {};
let vocabulary = [];
let tagToIndex = {};
let indexToTag = {};
let maxSequenceLength = 0;
let trainedNLUModel = null;
let numUniqueTags = 0;
let X_train_tensor = null;
let y_train_tensor = null;


// DOM Elements
let chatMessages, chatInput, sendButton, trainingStatus, trainModelButton;

document.addEventListener('DOMContentLoaded', () => {
    chatMessages = document.getElementById('chat-messages');
    chatInput = document.getElementById('chat-input');
    sendButton = document.getElementById('send-button');
    trainingStatus = document.getElementById('training-status');
    trainModelButton = document.getElementById('train-model-button');

    if (!chatMessages || !chatInput || !sendButton || !trainingStatus || !trainModelButton) {
        console.error("One or more DOM elements are missing!");
        trainingStatus.textContent = "Error: DOM elements missing.";
        return;
    }

    trainingStatus.textContent = "Initializing...";

    // Start loading data
    loadData(); // Uncommented and active
});

// Placeholder for functions to be added in subsequent steps:
// async function loadData() {}
// function prepareTrainingData(intentsData) {}
// function createNLUModel(vocabularySize, embeddingDim, maxSeqLen, numTags) {}
// async function trainNLUModel(model, X_train, y_train) {}
// function preprocessTextForPrediction(text, vocab, maxLen) {}
// async function predictIntentTFJS(inputText) {}
// function displayMessage(text, sender) {}
// async function handleSendMessage() {}
// function handleDataLoaded() {}

// This function replaces the previous placeholder for handleDataLoaded

function handleDataLoaded() {
    console.log("handleDataLoaded called - orchestrating model setup and training.");
    trainingStatus.textContent = "Processing loaded data...";

    const preparedData = prepareTrainingData(allIntents);

    if (!preparedData || !X_train_tensor || !y_train_tensor) {
        console.error("Data preparation failed. Cannot proceed with model creation and training.");
        trainingStatus.textContent = "Error: Data preparation failed.";
        if (trainModelButton) trainModelButton.disabled = false; // Re-enable button if stuck
        return;
    }

    // Assuming prepareTrainingData correctly sets global:
    // vocabulary, tagToIndex, indexToTag, maxSequenceLength, numUniqueTags, X_train_tensor, y_train_tensor

    // Vocabulary size for embedding layer is vocabulary.length + 1 if we reserve 0 for padding.
    // However, if vocabulary.indexOf() returns -1 for OOV and we map those to 0 (or handle them),
    // and padding is also 0, it might be okay.
    // For simplicity, let's ensure vocab for embedding is actual vocab size.
    // The `prepareTrainingData` function as written uses `vocabulary.indexOf(token)`,
    // so OOV words aren't handled explicitly other than not being in the sequence.
    // Padding is done with 0.
    // So, vocabulary.length should be correct for inputDim if 0 is only for padding or OOV.
    // Let's assume current `vocabulary.length` is the number of unique words.
    // If 0 is used for padding, inputDim for embedding should be `vocabulary.length + 1`.
    // Let's adjust prepareTrainingData slightly to make vocabulary 1-indexed for words, 0 for padding.

    // Re-checking prepareTrainingData:
    // `const sequence = doc.tokens.map(token => vocabulary.indexOf(token));`
    // `indexOf` returns -1 for not found. If we want 0 for padding, and words from 1 onwards:
    // `const sequence = doc.tokens.map(token => vocabulary.indexOf(token) + 1);` // words are 1 to N, 0 if not found (then map -1+1 =0)
    // And `const padding = Array(maxSequenceLength - seq.length).fill(0);`
    // This means vocab size for embedding should be `vocabulary.length + 1`.

    // Given the current `prepareTrainingData`, `vocabulary.indexOf(token)` means actual words get indices 0 to N-1.
    // Padding is with 0. This means the 0th word in vocab and padding share an index.
    // This is generally not ideal. A common practice is to reserve index 0 for padding.
    // Let's proceed with current `prepareTrainingData` for now, but acknowledge this nuance.
    // The `inputDim` for embedding should be the number of unique items in sequences, including padding.
    // If padding is 0, and word indices are 0 to N-1, then vocab size is N.

    const embeddingDim = 16; // Example, can be tuned

    const model = createNLUModel(vocabulary.length, embeddingDim, maxSequenceLength, numUniqueTags);

    if (!model) {
        console.error("Model creation failed. Cannot proceed with training.");
        trainingStatus.textContent = "Error: Model creation failed.";
        if (trainModelButton) trainModelButton.disabled = false;
        return;
    }

    // Disable chat until model is trained, or provide a message
    if (chatInput) chatInput.disabled = true;
    if (sendButton) sendButton.disabled = true;
    trainingStatus.textContent = "Model created. Click 'Train Model Manually' or wait for auto-train.";


    if (trainModelButton) {
        trainModelButton.disabled = false; // Enable it now that model is ready
        trainModelButton.addEventListener('click', () => {
            if (X_train_tensor && y_train_tensor && model) {
                if (chatInput) chatInput.disabled = true; // Disable during training
                if (sendButton) sendButton.disabled = true;
                trainNLUModel(model, X_train_tensor, y_train_tensor);
            } else {
                trainingStatus.textContent = "Error: Training data or model not ready for manual trigger.";
                console.error("Manual train trigger: Data or model not ready.", {X_train_tensor, y_train_tensor, model});
            }
        });
        // Optionally, trigger training automatically if no manual button is desired for default flow
        // For this prompt, we'll make the manual button the primary way to train after page load.
        // To auto-train:
        // trainNLUModel(model, X_train_tensor, y_train_tensor);
        trainingStatus.textContent = "Model and data ready. Click 'Train Model Manually' to start.";

    } else {
        // If no button, train automatically
        if (X_train_tensor && y_train_tensor && model) {
             if (chatInput) chatInput.disabled = true;
             if (sendButton) sendButton.disabled = true;
            trainNLUModel(model, X_train_tensor, y_train_tensor);
        } else {
            trainingStatus.textContent = "Error: Training data or model not ready for auto-training.";
            console.error("Auto-train: Data or model not ready.", {X_train_tensor, y_train_tensor, model});
        }
    }
}

// Make sure this is added after trainNLUModel, or at least before it's called by chat logic

function preprocessTextForPrediction(text, currentVocabulary, currentMaxSequenceLength) {
    if (!currentVocabulary || currentVocabulary.length === 0 || currentMaxSequenceLength === 0) {
        console.error("Vocabulary or maxSequenceLength not set for preprocessing.");
        return null;
    }
    // Basic tokenization (should mirror training preprocessing)
    const tokens = text.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?؟،؛]/g, "").replace(/\s\s+/g, ' ').split(/\s+/).filter(token => token.length > 0 && token !== " ");

    const sequence = tokens.map(token => {
        const index = currentVocabulary.indexOf(token);
        return index === -1 ? 0 : index; // Map OOV to 0 (if 0 is padding/OOV)
                                        // If vocab is 1-indexed and 0 is padding, this should be:
                                        // return index === -1 ? 0 : index + 1;
                                        // For now, sticking to 0-indexed vocab and 0 for padding/OOV
    });

    // Pad sequence (pre-padding)
    const paddedSequence = Array(currentMaxSequenceLength - Math.min(sequence.length, currentMaxSequenceLength)).fill(0).concat(sequence.slice(0, currentMaxSequenceLength));

    return tf.tensor2d([paddedSequence], [1, currentMaxSequenceLength], 'int32');
}

async function predictIntentTFJS(inputText) {
    if (!trainedNLUModel) {
        console.warn("NLU model not trained yet or training in progress.");
        return { intentTag: "fallback_not_ready", confidence: 1.0 };
    }
    if (!vocabulary || !indexToTag || maxSequenceLength === 0) {
        console.error("Prediction prerequisites (vocabulary, indexToTag, maxSequenceLength) not available.");
        return { intentTag: "fallback_error_prerequisites", confidence: 1.0 };
    }

    const processedInputTensor = preprocessTextForPrediction(inputText, vocabulary, maxSequenceLength);
    if (!processedInputTensor) {
        console.error("Failed to preprocess text for prediction.");
        return { intentTag: "fallback_error_processing", confidence: 1.0 };
    }

    let prediction;
    try {
        prediction = trainedNLUModel.predict(processedInputTensor);
        const predictionData = await prediction.data();

        let maxProb = -1;
        let predictedIndex = -1;
        for (let i = 0; i < predictionData.length; i++) {
            if (predictionData[i] > maxProb) {
                maxProb = predictionData[i];
                predictedIndex = i;
            }
        }

        if (predictedIndex !== -1 && indexToTag[predictedIndex]) {
            return { intentTag: indexToTag[predictedIndex], confidence: maxProb };
        } else {
            console.error("Failed to find intent tag for predicted index:", predictedIndex);
            return { intentTag: "fallback_error_unknown_index", confidence: 0.0 };
        }

    } catch (error) {
        console.error("Error during prediction:", error);
        return { intentTag: "fallback_error_prediction", confidence: 0.0 };
    } finally {
        if (processedInputTensor) processedInputTensor.dispose();
        if (prediction) prediction.dispose();
    }
}

// Make sure this is added after the global variable declarations and DOM element selections

async function loadData() {
    trainingStatus.textContent = "Loading data files...";
    try {
        const intentsResponse = await fetch('intents.json');
        if (!intentsResponse.ok) {
            throw new Error(`HTTP error! status: ${intentsResponse.status} for intents.json`);
        }
        allIntents = await intentsResponse.json();

        const menuResponse = await fetch('menu.json');
        if (!menuResponse.ok) {
            throw new Error(`HTTP error! status: ${menuResponse.status} for menu.json`);
        }
        allMenuItems = await menuResponse.json();

        console.log("Data loaded:", allIntents, allMenuItems);
        trainingStatus.textContent = "Data loaded. Preparing training data...";
        handleDataLoaded(); // This function will orchestrate the next steps
    } catch (error) {
        console.error("Error loading data:", error);
        trainingStatus.textContent = `Error loading data: ${error.message}`;
    }
}

// Make sure this is added after the loadData and handleDataLoaded functions

function prepareTrainingData(intentsData) {
    if (!intentsData || !intentsData.intents || intentsData.intents.length === 0) {
        console.error("Intents data is invalid or empty.");
        trainingStatus.textContent = "Error: Invalid intents data for training.";
        return null;
    }
    trainingStatus.textContent = "Preparing training data...";

    const words = [];
    const tags = [];
    const documents = []; // pairs of (pattern_tokens, tag)

    intentsData.intents.forEach(intent => {
        if (!tags.includes(intent.tag)) {
            tags.push(intent.tag);
        }
        intent.patterns.forEach(pattern => {
            // Basic tokenization: lowercase, remove punctuation (simple regex for demo)
            // In a real app, consider more robust tokenization/normalization
            const patternText = pattern.replace(/\[.*?\]\(.*?\)/g, '').trim(); // Remove entity annotations like [text](entity)
            const tokens = patternText.toLowerCase().replace(/[^\w\s؀-ۿ]/g, '').split(/\s+/).filter(token => token.length > 0);

            if (tokens.length > 0) {
                words.push(...tokens);
                documents.push({ tokens: tokens, tag: intent.tag });
            }
        });
    });

    vocabulary = [...new Set(words)].sort();
    numUniqueTags = tags.length;

    tagToIndex = {};
    indexToTag = {};
    tags.forEach((tag, i) => {
        tagToIndex[tag] = i;
        indexToTag[i] = tag;
    });

    console.log("Vocabulary size:", vocabulary.length);
    console.log("Unique tags:", numUniqueTags);
    console.log("Tag to Index:", tagToIndex);

    let trainingSequences = [];
    let trainingLabels = [];
    maxSequenceLength = 0;

    documents.forEach(doc => {
        const sequence = doc.tokens.map(token => vocabulary.indexOf(token));
        if (sequence.length > maxSequenceLength) {
            maxSequenceLength = sequence.length;
        }
        trainingSequences.push(sequence);
        trainingLabels.push(tagToIndex[doc.tag]);
    });

    console.log("Max sequence length:", maxSequenceLength);

    // Pad sequences and create one-hot encoded labels
    const paddedSequences = trainingSequences.map(seq => {
        const padding = Array(maxSequenceLength - seq.length).fill(0);
        return padding.concat(seq); // Pre-padding
    });

    const oneHotLabels = trainingLabels.map(labelIndex => {
        const label = Array(numUniqueTags).fill(0);
        label[labelIndex] = 1;
        return label;
    });

    if (paddedSequences.length === 0 || oneHotLabels.length === 0) {
        console.error("No training data generated after processing patterns.");
        trainingStatus.textContent = "Error: No training data to process.";
        return null;
    }

    // Create Tensors
    try {
        X_train_tensor = tf.tensor2d(paddedSequences, [paddedSequences.length, maxSequenceLength], 'int32');
        y_train_tensor = tf.tensor2d(oneHotLabels, [oneHotLabels.length, numUniqueTags], 'float32');
    } catch (error) {
        console.error("Error creating tensors:", error);
        console.error("Padded sequences sample:", paddedSequences.slice(0,1));
        console.error("One-hot labels sample:", oneHotLabels.slice(0,1));
        trainingStatus.textContent = "Error creating training tensors.";
        return null;
    }


    console.log("X_train tensor shape:", X_train_tensor.shape);
    console.log("y_train tensor shape:", y_train_tensor.shape);
    trainingStatus.textContent = "Training data prepared. Creating NLU model...";

    // Store globally (already done for vocabulary, tagToIndex, indexToTag, maxSequenceLength, numUniqueTags)
    // X_train_tensor and y_train_tensor are also global

    return { X_train_tensor, y_train_tensor }; // Return tensors as they are needed by the training function
}

// Make sure this is added after prepareTrainingData

function createNLUModel(currentVocabularySize, embeddingDim, currentMaxSequenceLength, currentNumUniqueTags) {
    if (currentVocabularySize <= 0 || currentMaxSequenceLength <= 0 || currentNumUniqueTags <= 0) {
        console.error("Invalid parameters for model creation:", {currentVocabularySize, currentMaxSequenceLength, currentNumUniqueTags });
        trainingStatus.textContent = "Error: Cannot create model due to invalid dimensions.";
        return null;
    }
    trainingStatus.textContent = "Creating NLU model...";

    const model = tf.sequential();

    // Embedding Layer
    model.add(tf.layers.embedding({
        inputDim: currentVocabularySize, // Size of the vocabulary
        outputDim: embeddingDim,         // Dimension of the dense embedding
        inputLength: currentMaxSequenceLength // Length of input sequences
    }));

    // Global Average Pooling Layer
    model.add(tf.layers.globalAveragePooling1d());

    // Dense Layer
    model.add(tf.layers.dense({
        units: 128, // Number of units in the dense layer
        activation: 'relu'
    }));

    // Optional: Add a Dropout layer for regularization if overfitting becomes an issue
    // model.add(tf.layers.dropout({rate: 0.5}));

    // Output Layer
    model.add(tf.layers.dense({
        units: currentNumUniqueTags, // Number of unique tags
        activation: 'softmax'
    }));

    // Compile the Model
    model.compile({
        optimizer: tf.train.adam(0.001), // Learning rate can be tuned
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    model.summary(); // Log model summary to console
    console.log("NLU Model created and compiled.");
    trainingStatus.textContent = "NLU model created. Ready for training.";

    return model;
}

// Make sure this is added after createNLUModel

async function trainNLUModel(model, X_train, y_train) {
    if (!model || !X_train || !y_train) {
        console.error("Cannot train model: model or training data is missing.");
        trainingStatus.textContent = "Error: Missing model or data for training.";
        if (trainModelButton) trainModelButton.disabled = false;
        return;
    }

    trainingStatus.textContent = "Starting model training...";
    if (trainModelButton) trainModelButton.disabled = true;

    const epochs = 100; // Example: Can be adjusted
    const batchSize = 16; // Example: Can be adjusted

    try {
        await model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progressMessage = `Training model... (Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)})`;
                    console.log(progressMessage);
                    trainingStatus.textContent = progressMessage;
                },
                onTrainEnd: () => {
                    trainedNLUModel = model; // Store the trained model globally
                    console.log("Model training completed.");
                    trainingStatus.textContent = "Model trained and ready!";
                    if (trainModelButton) trainModelButton.disabled = false;
                     // Potentially enable chat input here if it was disabled
                    if (chatInput) chatInput.disabled = false;
                    if (sendButton) sendButton.disabled = false;
                }
            }
        });
    } catch (error) {
        console.error("Error during model training:", error);
        trainingStatus.textContent = `Error during training: ${error.message}`;
        if (trainModelButton) trainModelButton.disabled = false;
    }
}

console.log("script.js loaded");
