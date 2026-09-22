Tweet Sentiment Classification: LSTM vs. Fine-Tuned BERT
Goal: compare a sequential neural network (LSTM) trained from scratch against a pretrained Transformer (DistilBERT) fine-tuned on the same task, to understand the practical tradeoffs between the two dominant architectures in modern NLP.
Data
Downloaded the Twitter US Airline Sentiment dataset from Kaggle (14,640 tweets) and loaded it into Google Colab using pandas. The two relevant fields are the raw tweet text and its labeled sentiment (positive, negative, or neutral).
Exploratory analysis revealed a significant class imbalance — 62.7% of tweets are negative, with neutral and positive making up the remainder. This became an important reference point: a naive model that always predicts "negative" would already score 62.7% accuracy without learning anything, so this became the baseline both real models needed to meaningfully outperform.
Preprocessing
Cleaned the raw text by removing @mentions (via regex) and lowercasing, then encoded the sentiment labels into integers (0/1/2) for compatibility with both models' loss functions.
Performed a single train/test split (80/20) on the raw text before any tokenization, to prevent data leakage — fitting a tokenizer's vocabulary on data that includes the test set can artificially inflate performance, since the model would be indirectly exposed to test-set vocabulary during preprocessing. Both models were trained and evaluated on the exact same split, ensuring a controlled, credible comparison.
Tokenization — Two Different Approaches
LSTM: used Keras' Tokenizer, which builds a vocabulary from the training corpus and assigns each word a unique integer ID (word-level tokenization). Sequences were padded/truncated to a fixed length of 15 tokens.
BERT: used Hugging Face's pretrained DistilBertTokenizer, which applies subword tokenization — breaking unfamiliar or complex words into smaller known sub-units. This allows BERT to handle out-of-vocabulary words gracefully, unlike a fixed word-level vocabulary. The tokenizer also generates an attention mask, indicating which tokens are real content versus padding.
Model 1: LSTM (Trained from Scratch)
Architecture: Embedding → LSTM → Dense (softmax)
Embedding layer: input_dim=7000 (vocabulary size), output_dim=64 (each word represented as a 64-dimensional learned vector), input_length=15
LSTM layer: 64 units, processes the sequence token-by-token, maintaining a hidden state to capture context over the sequence
Output layer: Dense(3, activation='softmax') for 3-class classification
Compiled with the Adam optimizer and sparse categorical crossentropy (appropriate since labels are integer-encoded rather than one-hot). Total trainable parameters: ~481K.
Model 2: Fine-Tuned DistilBERT
Loaded distilbert-base-uncased via Hugging Face's DistilBertForSequenceClassification, leveraging transfer learning — the model arrives already pretrained on a massive text corpus, so fine-tuning only adapts its existing language understanding to this specific classification task rather than learning language structure from zero. Fine-tuned using Hugging Face's Trainer API for 3 epochs with a small learning rate (3e-5), standard practice when fine-tuning pretrained models to avoid overwriting valuable pretrained weights. Total parameters: ~66M — roughly 137x larger than the LSTM.
Results
Model	Best Accuracy	Precision (weighted)	Recall (weighted)	Parameters
LSTM	78.2%	—	—	~481K
DistilBERT	79.9%	80.0%	79.9%	~66M
Baseline (always "negative")	62.7%	—	—	—
Both models comfortably beat the 62.7% baseline, confirming each learned genuine sentiment patterns rather than defaulting to the majority class.
Key Finding: Overfitting in Both Architectures
Despite their architectural differences, both models exhibited overfitting on this relatively small dataset, just at different rates:
The LSTM's validation accuracy peaked at epoch 2 (78.2%) before steadily declining as training accuracy climbed toward 95%+ — a classic sign of memorizing training-specific patterns rather than generalizing.
DistilBERT's validation loss was lowest at epoch 1, then increased through epochs 2 and 3 even as training loss kept dropping sharply — the same overfitting dynamic, compressed into far fewer epochs due to its much larger parameter count.
BERT outperformed the LSTM by only ~1.7 percentage points — a modest gap given it has roughly 137x more parameters. This suggests that on small, relatively simple datasets, both architectures converge toward a similar performance ceiling, and BERT's pretrained-knowledge advantage becomes more pronounced with larger, more complex datasets rather than being guaranteed regardless of data size.
Tools Used
Python, TensorFlow/Keras, PyTorch, Hugging Face Transformers, scikit-learn, pandas, Google Colab
Future Improvements
Apply early stopping or dropout to reduce overfitting in both models
Test on a larger dataset to see whether BERT's advantage over the LSTM widens
Evaluate both models on manually written adversarial examples (sarcasm, negation, mixed sentiment)
Compare training time/compute cost directly, not just accuracy, to quantify the practical tradeoff

