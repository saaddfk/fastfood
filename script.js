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

    // Attach event listeners for sending messages
    if (sendButton) {
        sendButton.addEventListener('click', handleSendMessage);
    }
    if (chatInput) {
        chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                // Optionally, prevent default form submission if input is in a form
                // event.preventDefault();
                handleSendMessage();
            }
        });
        // Initial state for chat input until model is trained
        chatInput.disabled = true;
        sendButton.disabled = true;
    }

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

    // MODIFIED PART: Map tokens to indices (0 for OOV/padding, 1 to N for words)
    const sequence = tokens.map(token => {
        const index = currentVocabulary.indexOf(token); // index is 0 to N-1 for known
        return index === -1 ? 0 : index + 1;   // map to 0 for OOV, or 1 to N for known
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
    trainingStatus.textContent = "Preparing training data (v2 indexing)...";

    const words = [];
    const tags = [];
    const documents = [];

    intentsData.intents.forEach(intent => {
        if (!tags.includes(intent.tag)) {
            tags.push(intent.tag);
        }
        intent.patterns.forEach(pattern => {
            const patternText = pattern.replace(/\[.*?\]\(.*?\)/g, '').trim();
            const tokens = patternText.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?؟،؛]/g, "").replace(/\s\s+/g, ' ').split(/\s+/).filter(token => token.length > 0 && token !== " ");

            if (tokens.length > 0) {
                words.push(...tokens);
                documents.push({ tokens: tokens, tag: intent.tag });
            }
        });
    });

    vocabulary = [...new Set(words)].sort(); // Vocabulary words (0 to N-1 initially)
    numUniqueTags = tags.length;

    tagToIndex = {};
    indexToTag = {};
    tags.forEach((tag, i) => {
        tagToIndex[tag] = i;
        indexToTag[i] = tag;
    });

    console.log("Vocabulary (0-indexed):", vocabulary.slice(0, 10)); // Display how vocab is stored
    console.log("Vocabulary size:", vocabulary.length);
    console.log("Unique tags:", numUniqueTags);

    let trainingSequences = [];
    let trainingLabels = [];
    maxSequenceLength = 0;

    documents.forEach(doc => {
        // MODIFIED PART: Map tokens to indices (0 for OOV/padding, 1 to N for words)
        const sequence = doc.tokens.map(token => {
            const index = vocabulary.indexOf(token); // index is 0 to N-1 for known words
            return index === -1 ? 0 : index + 1;   // map to 0 for OOV, or 1 to N for known
        });

        if (sequence.length > maxSequenceLength) {
            maxSequenceLength = sequence.length;
        }
        trainingSequences.push(sequence);
        trainingLabels.push(tagToIndex[doc.tag]);
    });

    console.log("Max sequence length:", maxSequenceLength);
    console.log("Sample sequence (1-indexed words, 0 for OOV/pad):", trainingSequences.length > 0 ? trainingSequences[0] : "N/A");


    // Pad sequences and create one-hot encoded labels
    const paddedSequences = trainingSequences.map(seq => {
        const paddingLength = maxSequenceLength - seq.length;
        const padding = Array(paddingLength).fill(0); // Padding is 0
        return padding.concat(seq);
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

    try {
        if (X_train_tensor) X_train_tensor.dispose(); // Dispose previous tensor if any
        X_train_tensor = tf.tensor2d(paddedSequences, [paddedSequences.length, maxSequenceLength], 'int32');

        if (y_train_tensor) y_train_tensor.dispose(); // Dispose previous tensor if any
        y_train_tensor = tf.tensor2d(oneHotLabels, [oneHotLabels.length, numUniqueTags], 'float32');
    } catch (error) {
        console.error("Error creating tensors:", error);
        trainingStatus.textContent = "Error creating training tensors.";
        return null;
    }

    console.log("X_train tensor (1-idx words) shape:", X_train_tensor.shape);
    console.log("y_train tensor shape:", y_train_tensor.shape);
    trainingStatus.textContent = "Training data prepared (v2 indexing). Creating NLU model...";

    return { X_train_tensor, y_train_tensor };
}

// Make sure this is added after prepareTrainingData

function createNLUModel(currentVocabularySize, embeddingDim, currentMaxSequenceLength, currentNumUniqueTags) {
    if (currentVocabularySize <= 0 || currentMaxSequenceLength <= 0 || currentNumUniqueTags <= 0) {
        console.error("Invalid parameters for model creation:", {currentVocabularySize, currentMaxSequenceLength, currentNumUniqueTags });
        trainingStatus.textContent = "Error: Cannot create model due to invalid dimensions.";
        return null;
    }
    trainingStatus.textContent = "Creating NLU model (v2 indexing)...";

    const model = tf.sequential();

    // MODIFIED PART: Embedding Layer inputDim
    model.add(tf.layers.embedding({
        inputDim: currentVocabularySize + 1, // Size of the vocabulary + 1 for padding/OOV at index 0
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
    console.log("NLU Model (v2 indexing) created and compiled.");
    trainingStatus.textContent = "NLU model (v2 indexing) created. Ready for training.";

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

// Make sure this is added before handleSendMessage or any other function that might use it.
// Typically, helper functions like this are defined earlier in the script or grouped together.

function displayMessage(text, sender) {
    if (!chatMessages) {
        console.error("Chat messages container not found!");
        return;
    }
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender); // sender will be 'user' or 'bot'

    // Sanitize text before setting it as textContent to prevent XSS if the source is ever not trusted.
    // For this application, text is either from user input (which is fine for textContent)
    // or from our JSON files (assumed to be safe).
    // If responses could contain HTML, more robust sanitization would be needed.
    messageDiv.textContent = text;

    chatMessages.appendChild(messageDiv);

    // Auto-scroll to the bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Make sure this is added after displayMessage, predictIntentTFJS, and global variables like allMenuItems, allIntents are defined.

async function handleSendMessage() {
    if (!chatInput || !chatMessages) {
        console.error("Chat input or messages container not found.");
        return;
    }
    const userText = chatInput.value.trim();
    if (userText === "") {
        return;
    }

    displayMessage(userText, 'user');
    chatInput.value = ""; // Clear input field

    // Disable input while bot is "thinking"
    chatInput.disabled = true;
    if (sendButton) sendButton.disabled = true;

    try {
        const prediction = await predictIntentTFJS(userText);
        let botResponse = "";
        const confidenceThreshold = 0.6; // Example threshold

        if (prediction && prediction.confidence > confidenceThreshold) {
            const intentTag = prediction.intentTag;
            const intentDetails = allIntents.intents.find(intent => intent.tag === intentTag);

            if (intentTag === 'ask_price' || intentTag === 'ask_ingredients') {
                let foundItemName = null;
                const userTextLower = userText.toLowerCase();
                for (const itemName in allMenuItems) {
                    if (userTextLower.includes(itemName.toLowerCase())) {
                        foundItemName = itemName;
                        break;
                    }
                }

                if (foundItemName && allMenuItems[foundItemName]) {
                    if (intentTag === 'ask_price') {
                        botResponse = `The price of ${foundItemName} is ${allMenuItems[foundItemName].price}.`;
                    } else { // ask_ingredients
                        const ingredients = allMenuItems[foundItemName].ingredients;
                        if (ingredients && ingredients.length > 0) {
                            botResponse = `The ingredients for ${foundItemName} are: ${ingredients.join(', ')}.`;
                        } else {
                            botResponse = `I don't have the ingredient information for ${foundItemName}.`;
                        }
                    }
                } else {
                    // Fallback to generic response for the intent if item not found
                    if (intentDetails && intentDetails.responses.length > 0) {
                        botResponse = intentDetails.responses[Math.floor(Math.random() * intentDetails.responses.length)];
                         // Add a note if it was looking for an item
                        if (intentTag === 'ask_price') botResponse += " Which item's price are you asking about?";
                        if (intentTag === 'ask_ingredients') botResponse += " Which item's ingredients are you asking about?";
                    } else {
                        botResponse = "I can help with prices and ingredients. Which item are you interested in?";
                    }
                }
            } else if (intentDetails && intentDetails.responses.length > 0) {
                // For other intents, select a random response
                botResponse = intentDetails.responses[Math.floor(Math.random() * intentDetails.responses.length)];
            } else {
                // Fallback if intent is recognized but has no specific responses defined (should not happen with good intents.json)
                botResponse = "I understood that, but I don't have a specific response prepared.";
            }
        } else if (prediction && prediction.intentTag === 'fallback_not_ready') {
             botResponse = "The NLU model is not ready yet. Please try again after it's trained.";
        } else if (prediction && prediction.intentTag && prediction.intentTag.startsWith('fallback_error')) {
            botResponse = "Sorry, I encountered an internal error trying to understand that.";
            console.error("Fallback error from predictIntentTFJS:", prediction.intentTag);
        }
        else {
            // Low confidence or other fallback
            botResponse = "I'm not quite sure how to respond to that. Can you try rephrasing, or ask about menu items?";
            if (prediction) { // Log if there was a prediction, even if low confidence
                 console.log(`Low confidence prediction: ${prediction.intentTag} (${prediction.confidence.toFixed(3)})`);
            }
        }

        displayMessage(botResponse, 'bot');

    } catch (error) {
        console.error("Error in handleSendMessage:", error);
        displayMessage("Sorry, something went wrong on my end.", 'bot');
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        if (sendButton) sendButton.disabled = false;
        chatInput.focus();
    }
}

console.log("script.js loaded");
