"""
Script to seed the knowledge base with educational content
Version 2.0 - Expanded with 30+ documents across multiple topics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import asyncio
from typing import List, Dict
from src.knowledge_base.vector_store import VectorStore
from src.knowledge_base.document_processor import DocumentProcessor
from src.agents.scraper import ScraperAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# COMPREHENSIVE EDUCATIONAL CONTENT (30+ Documents)
# =============================================================================

SAMPLE_DOCUMENTS = [
    # =========================================================================
    # MACHINE LEARNING FUNDAMENTALS (5 documents)
    # =========================================================================
    {
        "title": "Introduction to Machine Learning",
        "content": """Machine Learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Machine learning algorithms are categorized as supervised, unsupervised, semi-supervised, and reinforcement learning. Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data.

Key concepts in machine learning include:
- Training Data: The dataset used to train the model
- Features: Input variables used to make predictions
- Labels: The output variable we want to predict
- Model: The mathematical representation learned from data
- Overfitting: When a model learns the training data too well and performs poorly on new data
- Underfitting: When a model is too simple to capture patterns in the data""",
        "source": "educational",
        "topic": "machine-learning"
    },
    {
        "title": "Supervised Learning Explained",
        "content": """Supervised learning is a type of machine learning where the algorithm learns from labeled training data. In supervised learning, we have input variables (X) and an output variable (Y), and we use an algorithm to learn the mapping function from the input to the output: Y = f(X).

The goal is to approximate the mapping function so well that when we have new input data (X), we can predict the output variables (Y) for that data. It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process.

Types of Supervised Learning:
1. Classification: When the output variable is a category (e.g., spam/not spam, disease/no disease)
   - Binary Classification: Two classes
   - Multi-class Classification: More than two classes

2. Regression: When the output variable is a real value (e.g., price, temperature)
   - Linear Regression
   - Polynomial Regression

Common Supervised Learning Algorithms:
- Linear Regression: Predicts continuous values using a linear equation
- Logistic Regression: Used for binary classification problems
- Decision Trees: Tree-like model of decisions
- Random Forest: Ensemble of decision trees
- Support Vector Machines (SVM): Finds optimal hyperplane for classification
- K-Nearest Neighbors (KNN): Classifies based on closest training examples
- Naive Bayes: Probabilistic classifier based on Bayes theorem""",
        "source": "educational",
        "topic": "machine-learning"
    },
    {
        "title": "Unsupervised Learning Techniques",
        "content": """Unsupervised learning is a type of machine learning where the algorithm learns patterns from unlabeled data. Unlike supervised learning, there are no correct answers or labels to guide the learning process. The algorithm must find structure and patterns in the data on its own.

Types of Unsupervised Learning:

1. Clustering: Grouping similar data points together
   - K-Means Clustering: Partitions data into K clusters based on distance
   - Hierarchical Clustering: Creates a tree of clusters
   - DBSCAN: Density-based clustering that can find arbitrarily shaped clusters
   - Gaussian Mixture Models: Probabilistic model for clustering

2. Dimensionality Reduction: Reducing the number of features while preserving important information
   - Principal Component Analysis (PCA): Linear transformation to new coordinate system
   - t-SNE: Non-linear technique for visualization
   - Autoencoders: Neural networks for learning compressed representations

3. Association Rule Learning: Finding interesting relations between variables
   - Apriori Algorithm: Finds frequent itemsets
   - FP-Growth: More efficient than Apriori

Applications of Unsupervised Learning:
- Customer segmentation in marketing
- Anomaly detection in fraud prevention
- Image compression
- Topic modeling in text analysis
- Recommendation systems""",
        "source": "educational",
        "topic": "machine-learning"
    },
    {
        "title": "Model Evaluation and Validation",
        "content": """Model evaluation is crucial in machine learning to assess how well a model will perform on unseen data. Using the right evaluation metrics and validation techniques helps prevent overfitting and ensures the model generalizes well.

Evaluation Metrics for Classification:
- Accuracy: Percentage of correct predictions (not good for imbalanced data)
- Precision: True Positives / (True Positives + False Positives)
- Recall (Sensitivity): True Positives / (True Positives + False Negatives)
- F1 Score: Harmonic mean of precision and recall
- ROC-AUC: Area under the Receiver Operating Characteristic curve
- Confusion Matrix: Table showing TP, TN, FP, FN

Evaluation Metrics for Regression:
- Mean Squared Error (MSE): Average of squared differences
- Root Mean Squared Error (RMSE): Square root of MSE
- Mean Absolute Error (MAE): Average of absolute differences
- R-squared (R²): Proportion of variance explained by the model

Validation Techniques:
1. Train-Test Split: Simple split (e.g., 80% train, 20% test)
2. K-Fold Cross Validation: Split data into K folds, train on K-1, test on 1
3. Stratified K-Fold: Maintains class distribution in each fold
4. Leave-One-Out: Each sample is used once as test set
5. Time Series Split: For temporal data, maintains time order

Preventing Overfitting:
- Use more training data
- Regularization (L1/L2)
- Early stopping
- Dropout (for neural networks)
- Cross-validation for hyperparameter tuning""",
        "source": "educational",
        "topic": "machine-learning"
    },
    {
        "title": "Feature Engineering and Selection",
        "content": """Feature engineering is the process of using domain knowledge to create, transform, or select features (input variables) that improve machine learning model performance. Good features can dramatically improve model accuracy.

Feature Creation Techniques:
1. Mathematical Transformations:
   - Log transformation for skewed data
   - Square root or power transformations
   - Binning continuous variables

2. Date/Time Features:
   - Extract year, month, day, hour
   - Day of week, is_weekend
   - Time since an event

3. Text Features:
   - Bag of Words
   - TF-IDF (Term Frequency-Inverse Document Frequency)
   - Word embeddings (Word2Vec, GloVe)

4. Categorical Encoding:
   - One-Hot Encoding: Creates binary columns for each category
   - Label Encoding: Assigns integers to categories
   - Target Encoding: Uses target mean for each category

Feature Selection Methods:
1. Filter Methods: Based on statistical measures
   - Correlation with target
   - Chi-square test
   - Information gain

2. Wrapper Methods: Use model performance
   - Forward Selection
   - Backward Elimination
   - Recursive Feature Elimination (RFE)

3. Embedded Methods: Built into algorithms
   - Lasso Regression (L1)
   - Random Forest feature importance
   - Gradient Boosting feature importance

Feature Scaling:
- Standardization (Z-score): Mean=0, Std=1
- Min-Max Normalization: Scale to [0,1]
- Robust Scaling: Uses median and IQR""",
        "source": "educational",
        "topic": "machine-learning"
    },

    # =========================================================================
    # DEEP LEARNING (6 documents)
    # =========================================================================
    {
        "title": "Neural Networks Fundamentals",
        "content": """Artificial Neural Networks (ANNs) are computing systems inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain.

Components of a Neural Network:
1. Input Layer: Receives the input features
2. Hidden Layers: Process information through weighted connections
3. Output Layer: Produces the final prediction
4. Weights: Parameters that are learned during training
5. Biases: Additional parameters for flexibility
6. Activation Functions: Introduce non-linearity

Common Activation Functions:
- Sigmoid: σ(x) = 1/(1+e^(-x)), output between 0 and 1
- Tanh: Output between -1 and 1
- ReLU: max(0, x), most commonly used
- Leaky ReLU: Allows small negative values
- Softmax: For multi-class classification output

Training Neural Networks:
1. Forward Propagation: Calculate output from input
2. Loss Calculation: Measure difference from expected output
3. Backpropagation: Calculate gradients using chain rule
4. Weight Update: Adjust weights using gradient descent

Key Concepts:
- Learning Rate: Step size for weight updates
- Epochs: Number of complete passes through training data
- Batch Size: Number of samples per gradient update
- Loss Function: Measures prediction error (MSE, Cross-Entropy)""",
        "source": "educational",
        "topic": "deep-learning"
    },
    {
        "title": "Convolutional Neural Networks (CNN)",
        "content": """Convolutional Neural Networks (CNNs) are a class of deep neural networks most commonly applied to analyzing visual imagery. They are specifically designed to process pixel data and are highly effective for image classification, object detection, and image segmentation.

Key Components of CNN:

1. Convolutional Layer:
   - Applies filters (kernels) to input
   - Extracts features like edges, textures, patterns
   - Parameters: kernel size, number of filters, stride, padding
   - Produces feature maps

2. Pooling Layer:
   - Reduces spatial dimensions
   - Max Pooling: Takes maximum value in window
   - Average Pooling: Takes average value in window
   - Reduces computation and prevents overfitting

3. Fully Connected Layer:
   - Traditional neural network layer
   - Used at the end for classification

CNN Architecture Concepts:
- Receptive Field: Region of input affecting a neuron
- Stride: Step size when sliding the filter
- Padding: Adding zeros around input borders
- Feature Maps: Output of convolution operations

Famous CNN Architectures:
- LeNet-5: Early CNN for digit recognition
- AlexNet: Won ImageNet 2012, introduced ReLU and Dropout
- VGGNet: Deep network with small 3x3 filters
- ResNet: Introduced skip connections for very deep networks
- Inception: Multi-scale feature extraction

Applications:
- Image Classification
- Object Detection (YOLO, Faster R-CNN)
- Image Segmentation (U-Net)
- Face Recognition
- Medical Image Analysis""",
        "source": "educational",
        "topic": "deep-learning"
    },
    {
        "title": "Recurrent Neural Networks (RNN)",
        "content": """Recurrent Neural Networks (RNNs) are a class of neural networks designed for sequential data. Unlike feedforward networks, RNNs have connections that form directed cycles, allowing them to maintain a 'memory' of previous inputs.

How RNNs Work:
- Process sequences one element at a time
- Maintain hidden state that captures information from past
- Same weights are shared across all time steps
- Output at each step depends on current input and hidden state

Challenges with Basic RNNs:
1. Vanishing Gradient Problem:
   - Gradients become very small over long sequences
   - Network can't learn long-term dependencies

2. Exploding Gradient Problem:
   - Gradients become very large
   - Training becomes unstable

Solutions - Advanced RNN Architectures:

1. Long Short-Term Memory (LSTM):
   - Introduced by Hochreiter & Schmidhuber (1997)
   - Uses gates to control information flow:
     * Forget Gate: What to discard from cell state
     * Input Gate: What new information to add
     * Output Gate: What to output
   - Can learn long-term dependencies

2. Gated Recurrent Unit (GRU):
   - Simplified version of LSTM
   - Uses reset and update gates
   - Fewer parameters, often similar performance

Applications of RNNs:
- Language Modeling
- Machine Translation
- Speech Recognition
- Time Series Prediction
- Sentiment Analysis
- Text Generation""",
        "source": "educational",
        "topic": "deep-learning"
    },
    {
        "title": "Transformer Architecture",
        "content": """The Transformer architecture, introduced in the paper 'Attention Is All You Need' (2017), revolutionized natural language processing and has since been applied to many other domains. Unlike RNNs, Transformers process entire sequences in parallel using self-attention mechanisms.

Key Components:

1. Self-Attention Mechanism:
   - Allows each position to attend to all positions
   - Computes Query (Q), Key (K), and Value (V) matrices
   - Attention(Q,K,V) = softmax(QK^T/√d_k)V
   - Captures relationships between all words in a sequence

2. Multi-Head Attention:
   - Multiple attention mechanisms in parallel
   - Each head can focus on different aspects
   - Outputs are concatenated and projected

3. Position Encoding:
   - Since no recurrence, position info must be added
   - Uses sine and cosine functions of different frequencies
   - Allows model to understand word order

4. Feed-Forward Networks:
   - Applied to each position independently
   - Two linear transformations with ReLU

5. Layer Normalization:
   - Normalizes across features for each sample
   - Helps with training stability

Transformer Architecture:
- Encoder: Processes input sequence
- Decoder: Generates output sequence
- Both use stacks of identical layers

Advantages over RNNs:
- Parallel processing (faster training)
- Better at capturing long-range dependencies
- More scalable to large datasets

Famous Transformer Models:
- BERT: Bidirectional encoder
- GPT: Autoregressive decoder
- T5: Text-to-text framework""",
        "source": "educational",
        "topic": "deep-learning"
    },
    {
        "title": "Training Deep Neural Networks",
        "content": """Training deep neural networks effectively requires understanding various optimization techniques, regularization methods, and best practices. This guide covers essential concepts for successful deep learning.

Optimization Algorithms:

1. Stochastic Gradient Descent (SGD):
   - Updates weights using gradient of mini-batch
   - Can be slow and get stuck in local minima

2. SGD with Momentum:
   - Accumulates velocity in consistent direction
   - Helps overcome local minima

3. Adam (Adaptive Moment Estimation):
   - Combines momentum and adaptive learning rates
   - Most commonly used optimizer
   - Works well out of the box

4. RMSprop:
   - Adapts learning rate per parameter
   - Good for RNNs

Regularization Techniques:

1. L1/L2 Regularization:
   - Adds penalty for large weights
   - L1 promotes sparsity
   - L2 (weight decay) prevents large weights

2. Dropout:
   - Randomly sets neurons to zero during training
   - Prevents co-adaptation
   - Typically 0.2-0.5 dropout rate

3. Batch Normalization:
   - Normalizes layer inputs
   - Allows higher learning rates
   - Reduces internal covariate shift

4. Early Stopping:
   - Stop training when validation loss increases
   - Prevents overfitting

Learning Rate Scheduling:
- Step Decay: Reduce by factor at specific epochs
- Exponential Decay: Gradual reduction
- Cosine Annealing: Follows cosine curve
- Warm Restarts: Reset learning rate periodically

Best Practices:
- Start with established architectures
- Use pretrained models when possible
- Monitor training and validation loss
- Use appropriate batch size
- Initialize weights properly (Xavier, He)""",
        "source": "educational",
        "topic": "deep-learning"
    },
    {
        "title": "Transfer Learning and Pre-trained Models",
        "content": """Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different task. This is especially powerful in deep learning where training from scratch requires massive datasets and computational resources.

Why Transfer Learning Works:
- Early layers learn general features (edges, textures)
- Later layers learn task-specific features
- General features transfer well across tasks
- Reduces training time and data requirements

Transfer Learning Strategies:

1. Feature Extraction:
   - Use pre-trained model as fixed feature extractor
   - Only train new classification layers
   - Good when: new dataset is small, similar to original

2. Fine-Tuning:
   - Unfreeze some layers of pre-trained model
   - Train entire network with small learning rate
   - Good when: new dataset is larger, somewhat different

3. Progressive Fine-Tuning:
   - Gradually unfreeze layers from top to bottom
   - Allows careful adaptation

Pre-trained Models for Different Domains:

Computer Vision:
- ImageNet models: ResNet, VGG, EfficientNet
- Object Detection: YOLO, Faster R-CNN
- Segmentation: U-Net, DeepLab

Natural Language Processing:
- BERT: Bidirectional understanding
- GPT: Text generation
- RoBERTa: Optimized BERT
- T5: Text-to-text

Best Practices:
1. Choose model trained on similar domain
2. Freeze early layers initially
3. Use smaller learning rate for pre-trained layers
4. Data augmentation still helps
5. Monitor for overfitting

Fine-Tuning Tips:
- New data similar: freeze more layers
- New data different: unfreeze more layers
- Small new dataset: more regularization
- Large new dataset: can fine-tune aggressively""",
        "source": "educational",
        "topic": "deep-learning"
    },

    # =========================================================================
    # NATURAL LANGUAGE PROCESSING (5 documents)
    # =========================================================================
    {
        "title": "Natural Language Processing Basics",
        "content": """Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.

Core NLP Tasks:

1. Text Preprocessing:
   - Tokenization: Breaking text into words or subwords
   - Lowercasing: Converting to lowercase
   - Stop word removal: Removing common words (the, is, at)
   - Stemming: Reducing words to root form (running -> run)
   - Lemmatization: Converting to dictionary form

2. Part-of-Speech (POS) Tagging:
   - Identifying nouns, verbs, adjectives, etc.
   - Important for understanding sentence structure

3. Named Entity Recognition (NER):
   - Identifying entities: people, organizations, locations
   - Important for information extraction

4. Sentiment Analysis:
   - Determining opinion: positive, negative, neutral
   - Applications in social media monitoring, reviews

5. Text Classification:
   - Categorizing documents into predefined classes
   - Spam detection, topic classification

Text Representation:
- Bag of Words: Count of each word
- TF-IDF: Term frequency weighted by inverse document frequency
- Word Embeddings: Dense vector representations
- Contextual Embeddings: Context-dependent representations (BERT)""",
        "source": "educational",
        "topic": "nlp"
    },
    {
        "title": "Word Embeddings and Word2Vec",
        "content": """Word embeddings are dense vector representations of words where semantically similar words are mapped to nearby points in vector space. They capture semantic and syntactic relationships between words.

Why Word Embeddings:
- Overcome limitations of one-hot encoding
- Capture semantic relationships
- Reduce dimensionality
- Enable mathematical operations on words

Word2Vec (Google, 2013):
Two architectures for learning embeddings:

1. Continuous Bag of Words (CBOW):
   - Predicts target word from context words
   - Input: surrounding words
   - Output: center word
   - Faster training, good for frequent words

2. Skip-gram:
   - Predicts context words from target word
   - Input: center word
   - Output: surrounding words
   - Better for rare words and small datasets

Training Details:
- Negative Sampling: Efficient approximation
- Window Size: Number of context words
- Embedding Dimension: Typically 100-300

Word Embedding Properties:
- Similar words cluster together
- Analogies: king - man + woman ≈ queen
- Captures multiple relationships

Other Embedding Methods:
- GloVe: Global Vectors, uses co-occurrence matrix
- FastText: Includes subword information
- ELMo: Context-dependent embeddings

Limitations:
- Static embeddings: same vector regardless of context
- Solved by contextual embeddings (BERT, GPT)

Applications:
- Text similarity
- Document classification
- Named entity recognition
- Machine translation""",
        "source": "educational",
        "topic": "nlp"
    },
    {
        "title": "BERT and Modern Language Models",
        "content": """BERT (Bidirectional Encoder Representations from Transformers) is a language model developed by Google that revolutionized NLP by introducing bidirectional context understanding. Released in 2018, it set new benchmarks on many NLP tasks.

Key Innovations of BERT:

1. Bidirectional Context:
   - Unlike previous models that read left-to-right
   - BERT considers both left and right context
   - Better understanding of word meaning

2. Pre-training Tasks:
   a) Masked Language Modeling (MLM):
      - Randomly mask 15% of tokens
      - Model predicts masked tokens
      - Forces deep bidirectional learning

   b) Next Sentence Prediction (NSP):
      - Predict if sentence B follows sentence A
      - Helps understand sentence relationships

3. Fine-tuning:
   - Pre-trained on large corpus (Wikipedia, Books)
   - Fine-tune on specific tasks with small datasets
   - Add task-specific output layer

BERT Architecture:
- Transformer encoder (no decoder)
- BERT-Base: 12 layers, 768 hidden, 110M parameters
- BERT-Large: 24 layers, 1024 hidden, 340M parameters

BERT Variants:
- RoBERTa: More training data, no NSP
- DistilBERT: Smaller, faster (66M parameters)
- ALBERT: Parameter sharing for efficiency
- SpanBERT: Better span prediction
- BioBERT: Pre-trained on biomedical text

Using BERT:
- For classification: [CLS] token output
- For token tasks: Individual token outputs
- Libraries: Hugging Face Transformers

Applications:
- Question Answering
- Named Entity Recognition
- Sentiment Analysis
- Text Classification
- Semantic Search""",
        "source": "educational",
        "topic": "nlp"
    },
    {
        "title": "GPT and Text Generation",
        "content": """GPT (Generative Pre-trained Transformer) is a family of large language models developed by OpenAI. Unlike BERT's bidirectional approach, GPT uses unidirectional (left-to-right) attention for text generation.

GPT Architecture:
- Transformer decoder only
- Autoregressive: predicts next token
- Unidirectional attention (causal masking)
- Pre-trained on diverse internet text

GPT Evolution:
1. GPT-1 (2018): 117M parameters
2. GPT-2 (2019): 1.5B parameters
3. GPT-3 (2020): 175B parameters
4. GPT-4 (2023): Multimodal capabilities

Key Concepts:

1. Autoregressive Generation:
   - Predicts one token at a time
   - Uses previously generated tokens as context
   - P(text) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...

2. In-Context Learning:
   - Learn from examples in the prompt
   - No gradient updates needed
   - Zero-shot, Few-shot learning

3. Prompt Engineering:
   - Crafting effective prompts
   - Instructions, examples, context
   - Critical for good results

Generation Parameters:
- Temperature: Controls randomness (0=deterministic)
- Top-k: Sample from top k tokens
- Top-p (Nucleus): Sample from top cumulative probability
- Max tokens: Limit output length

Applications:
- Text Generation
- Code Generation
- Summarization
- Translation
- Question Answering
- Creative Writing
- Chatbots

Challenges:
- Hallucination: Generating false information
- Bias: Reflecting training data biases
- Computational cost: Massive resources needed""",
        "source": "educational",
        "topic": "nlp"
    },
    {
        "title": "Text Classification and Sentiment Analysis",
        "content": """Text classification is the task of assigning predefined categories to text documents. Sentiment analysis is a specific type of text classification that determines the emotional tone or opinion expressed in text.

Text Classification Approaches:

1. Traditional Machine Learning:
   - Feature extraction (TF-IDF, Bag of Words)
   - Classifiers: Naive Bayes, SVM, Random Forest
   - Good for smaller datasets

2. Deep Learning:
   - CNN for text: Captures local patterns
   - RNN/LSTM: Captures sequential dependencies
   - Transformers: State-of-the-art performance

3. Transfer Learning:
   - Fine-tune BERT, RoBERTa
   - Usually best performance
   - Works well with limited data

Sentiment Analysis Types:

1. Binary Sentiment:
   - Positive or Negative
   - Most common application

2. Multi-class Sentiment:
   - Positive, Neutral, Negative
   - Or fine-grained (1-5 stars)

3. Aspect-based Sentiment:
   - Sentiment toward specific aspects
   - "Food was great but service was slow"

4. Emotion Detection:
   - Joy, sadness, anger, fear, etc.
   - More nuanced than sentiment

Implementation Steps:
1. Data Collection and Labeling
2. Text Preprocessing
3. Feature Extraction or Tokenization
4. Model Training
5. Evaluation (Accuracy, F1, Confusion Matrix)
6. Deployment

Challenges:
- Sarcasm and irony detection
- Context-dependent meaning
- Negation handling
- Domain-specific vocabulary
- Class imbalance

Best Practices:
- Balance your dataset
- Use domain-specific pre-training
- Consider ensemble methods
- Evaluate on diverse test sets""",
        "source": "educational",
        "topic": "nlp"
    },

    # =========================================================================
    # PYTHON PROGRAMMING (4 documents)
    # =========================================================================
    {
        "title": "Python Basics for Data Science",
        "content": """Python is the most popular programming language for data science and machine learning due to its simplicity, readability, and powerful ecosystem of libraries. This guide covers essential Python concepts for data science.

Data Types:
- Numbers: int, float, complex
- Strings: Text data, immutable
- Lists: Ordered, mutable collection
- Tuples: Ordered, immutable collection
- Dictionaries: Key-value pairs
- Sets: Unordered, unique elements

Essential Operations:
- List comprehensions: [x*2 for x in range(10)]
- Dictionary comprehensions: {k: v*2 for k, v in d.items()}
- Lambda functions: lambda x: x*2
- Map, Filter, Reduce: Functional programming

Control Flow:
- if/elif/else for conditionals
- for loops for iteration
- while loops for condition-based iteration
- try/except for error handling

Functions:
- def keyword to define
- *args for variable positional arguments
- **kwargs for variable keyword arguments
- Return multiple values as tuple
- Docstrings for documentation

Object-Oriented Programming:
- Classes and objects
- Inheritance
- Encapsulation
- Methods and attributes

File Operations:
- open() for file handling
- with statement for context management
- Reading: read(), readline(), readlines()
- Writing: write(), writelines()

Important Libraries:
- NumPy: Numerical computing
- Pandas: Data manipulation
- Matplotlib/Seaborn: Visualization
- Scikit-learn: Machine learning
- TensorFlow/PyTorch: Deep learning""",
        "source": "educational",
        "topic": "python"
    },
    {
        "title": "NumPy for Numerical Computing",
        "content": """NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on them efficiently.

NumPy Arrays:
- ndarray: N-dimensional array object
- Homogeneous: All elements same type
- Fixed size at creation
- Much faster than Python lists

Creating Arrays:
- np.array([1, 2, 3]): From list
- np.zeros((3, 4)): Array of zeros
- np.ones((2, 3)): Array of ones
- np.arange(0, 10, 2): Range with step
- np.linspace(0, 1, 5): Evenly spaced
- np.random.rand(3, 3): Random values

Array Attributes:
- shape: Dimensions of array
- dtype: Data type of elements
- ndim: Number of dimensions
- size: Total number of elements

Array Operations:
- Element-wise: +, -, *, /, **
- Broadcasting: Operations on different shapes
- Dot product: np.dot(a, b) or a @ b
- Matrix operations: np.matmul, np.linalg

Indexing and Slicing:
- arr[0]: First element
- arr[-1]: Last element
- arr[1:4]: Slice
- arr[:, 0]: All rows, first column
- arr[arr > 5]: Boolean indexing

Reshaping:
- reshape(): Change dimensions
- flatten(): 1D array
- transpose(): Swap axes

Aggregations:
- sum(), mean(), std(), min(), max()
- argmin(), argmax(): Index of min/max
- cumsum(): Cumulative sum

Linear Algebra:
- np.linalg.inv(): Matrix inverse
- np.linalg.det(): Determinant
- np.linalg.eig(): Eigenvalues
- np.linalg.svd(): SVD decomposition""",
        "source": "educational",
        "topic": "python"
    },
    {
        "title": "Pandas for Data Analysis",
        "content": """Pandas is a powerful Python library for data manipulation and analysis. It provides data structures like DataFrame and Series that make working with structured data intuitive and efficient.

Core Data Structures:

1. Series:
   - 1D labeled array
   - Can hold any data type
   - Index for labels

2. DataFrame:
   - 2D labeled data structure
   - Like a spreadsheet or SQL table
   - Columns can have different types

Creating DataFrames:
- pd.DataFrame(dict): From dictionary
- pd.read_csv('file.csv'): From CSV
- pd.read_excel('file.xlsx'): From Excel
- pd.read_sql(query, conn): From database

Viewing Data:
- df.head(), df.tail(): First/last rows
- df.info(): Data types and memory
- df.describe(): Statistical summary
- df.shape: Dimensions
- df.columns: Column names

Selection:
- df['column']: Single column
- df[['col1', 'col2']]: Multiple columns
- df.loc[row, col]: Label-based
- df.iloc[row, col]: Integer-based
- df.query('col > 5'): Query string

Filtering:
- df[df['col'] > 5]: Boolean indexing
- df[(df['a'] > 1) & (df['b'] < 5)]: Multiple conditions

Data Cleaning:
- df.dropna(): Remove missing values
- df.fillna(value): Fill missing values
- df.drop_duplicates(): Remove duplicates
- df.replace(): Replace values
- df.rename(): Rename columns

Grouping and Aggregation:
- df.groupby('col').mean()
- df.groupby(['a', 'b']).agg({'c': 'sum', 'd': 'mean'})
- df.pivot_table(): Pivot tables

Merging:
- pd.merge(df1, df2, on='key'): SQL-like join
- pd.concat([df1, df2]): Concatenate""",
        "source": "educational",
        "topic": "python"
    },
    {
        "title": "Data Visualization with Matplotlib and Seaborn",
        "content": """Data visualization is essential for understanding data, communicating insights, and presenting findings. Matplotlib and Seaborn are Python's primary visualization libraries.

Matplotlib Basics:
- plt.figure(): Create new figure
- plt.plot(): Line plot
- plt.scatter(): Scatter plot
- plt.bar(): Bar chart
- plt.hist(): Histogram
- plt.show(): Display plot
- plt.savefig(): Save to file

Customization:
- plt.title(): Add title
- plt.xlabel(), plt.ylabel(): Axis labels
- plt.legend(): Add legend
- plt.xlim(), plt.ylim(): Axis limits
- plt.grid(): Add grid

Subplots:
- fig, axes = plt.subplots(2, 2)
- axes[0, 0].plot(x, y)
- Arrange multiple plots

Seaborn (Statistical Visualization):
Built on Matplotlib, provides higher-level interface

Distribution Plots:
- sns.histplot(): Histogram
- sns.kdeplot(): Kernel density
- sns.boxplot(): Box plot
- sns.violinplot(): Violin plot

Categorical Plots:
- sns.countplot(): Count of categories
- sns.barplot(): Bar with error bars
- sns.stripplot(): Points by category

Relationship Plots:
- sns.scatterplot(): Scatter with style
- sns.lineplot(): Line with confidence
- sns.regplot(): Regression plot

Matrix Plots:
- sns.heatmap(): Heatmap
- sns.clustermap(): Clustered heatmap

Multi-plot Grids:
- sns.FacetGrid(): Grid by categories
- sns.pairplot(): All pairwise relationships

Styling:
- sns.set_style('whitegrid'): Set style
- sns.set_palette('husl'): Set colors
- plt.style.use('seaborn'): Matplotlib style

Best Practices:
- Choose appropriate chart type
- Label everything clearly
- Use color meaningfully
- Avoid clutter
- Consider colorblind-friendly palettes""",
        "source": "educational",
        "topic": "python"
    },

    # =========================================================================
    # STATISTICS AND MATHEMATICS (5 documents)
    # =========================================================================
    {
        "title": "Probability Fundamentals",
        "content": """Probability is the mathematical study of uncertainty and randomness. Understanding probability is essential for machine learning, as many algorithms are based on probabilistic foundations.

Basic Concepts:
- Experiment: A process that produces outcomes
- Sample Space (S): Set of all possible outcomes
- Event: A subset of the sample space
- Probability: P(A) = number of favorable outcomes / total outcomes

Probability Rules:
- 0 ≤ P(A) ≤ 1
- P(S) = 1 (certain event)
- P(∅) = 0 (impossible event)
- P(A') = 1 - P(A) (complement)

Addition Rule:
- P(A or B) = P(A) + P(B) - P(A and B)
- If mutually exclusive: P(A or B) = P(A) + P(B)

Multiplication Rule:
- P(A and B) = P(A) × P(B|A)
- If independent: P(A and B) = P(A) × P(B)

Conditional Probability:
- P(A|B) = P(A and B) / P(B)
- Probability of A given B has occurred

Bayes' Theorem:
- P(A|B) = P(B|A) × P(A) / P(B)
- Foundation of Bayesian inference
- Used in Naive Bayes classifier

Random Variables:
- Discrete: Countable values (dice roll)
- Continuous: Any value in range (height)

Probability Distributions:
- PMF (Probability Mass Function): Discrete
- PDF (Probability Density Function): Continuous
- CDF (Cumulative Distribution Function): P(X ≤ x)

Common Distributions:
- Bernoulli: Single binary trial
- Binomial: n binary trials
- Poisson: Count of events
- Normal (Gaussian): Bell curve
- Uniform: Equal probability""",
        "source": "educational",
        "topic": "statistics"
    },
    {
        "title": "Statistical Distributions",
        "content": """Statistical distributions describe how data is spread or distributed. Understanding distributions is crucial for selecting appropriate models and making statistical inferences.

Discrete Distributions:

1. Bernoulli Distribution:
   - Single trial with success probability p
   - X ∈ {0, 1}
   - Mean: p, Variance: p(1-p)

2. Binomial Distribution:
   - n independent Bernoulli trials
   - X = number of successes
   - Mean: np, Variance: np(1-p)

3. Poisson Distribution:
   - Count of events in fixed interval
   - Parameter: λ (rate)
   - Mean: λ, Variance: λ

4. Geometric Distribution:
   - Trials until first success
   - Mean: 1/p

Continuous Distributions:

1. Normal (Gaussian) Distribution:
   - Bell-shaped curve
   - Parameters: μ (mean), σ (std dev)
   - 68-95-99.7 rule
   - Many natural phenomena

2. Uniform Distribution:
   - Equal probability over interval [a, b]
   - Mean: (a+b)/2

3. Exponential Distribution:
   - Time between Poisson events
   - Parameter: λ (rate)
   - Memoryless property

4. Chi-Square Distribution:
   - Sum of squared normal variables
   - Used in hypothesis testing

5. t-Distribution:
   - Similar to normal, heavier tails
   - Used for small samples
   - Approaches normal as df increases

6. F-Distribution:
   - Ratio of chi-squares
   - Used in ANOVA

Central Limit Theorem:
- Sample means → Normal distribution
- As sample size increases
- Regardless of population distribution
- Foundation of statistical inference""",
        "source": "educational",
        "topic": "statistics"
    },
    {
        "title": "Hypothesis Testing",
        "content": """Hypothesis testing is a statistical method for making decisions about population parameters based on sample data. It's fundamental to scientific research and data analysis.

Key Concepts:

1. Null Hypothesis (H₀):
   - Statement of no effect/difference
   - What we assume is true initially

2. Alternative Hypothesis (H₁ or Hₐ):
   - Statement of effect/difference
   - What we're trying to prove

3. Test Statistic:
   - Calculated from sample data
   - Measures how far sample is from H₀

4. P-value:
   - Probability of observing result if H₀ true
   - Smaller p-value = stronger evidence against H₀

5. Significance Level (α):
   - Threshold for rejecting H₀
   - Common values: 0.05, 0.01

Decision Rule:
- If p-value < α: Reject H₀
- If p-value ≥ α: Fail to reject H₀

Types of Errors:
- Type I (False Positive): Reject H₀ when true, probability = α
- Type II (False Negative): Fail to reject H₀ when false, probability = β
- Power = 1 - β: Probability of correctly rejecting false H₀

Common Tests:

1. Z-test: Known population variance, large samples
2. t-test: Unknown variance, small samples
   - One-sample: Compare mean to value
   - Two-sample: Compare two means
   - Paired: Compare paired observations

3. Chi-square test: Categorical data
   - Goodness of fit
   - Independence test

4. ANOVA: Compare multiple means
5. F-test: Compare variances

Assumptions:
- Random sampling
- Independence
- Normality (for many tests)
- Homogeneity of variance""",
        "source": "educational",
        "topic": "statistics"
    },
    {
        "title": "Linear Algebra for Machine Learning",
        "content": """Linear algebra is the mathematical foundation of machine learning. Most ML algorithms are expressed in terms of matrix and vector operations, making linear algebra essential for understanding how these algorithms work.

Vectors:
- Ordered list of numbers
- Column vector: n×1 matrix
- Operations: addition, scalar multiplication
- Dot product: a·b = Σaᵢbᵢ
- Magnitude: ||a|| = √(Σaᵢ²)
- Unit vector: ||a|| = 1

Matrices:
- 2D array of numbers (m×n)
- Identity matrix: I (diagonal of 1s)
- Transpose: Aᵀ (rows ↔ columns)
- Symmetric: A = Aᵀ

Matrix Operations:
- Addition: Element-wise (same dimensions)
- Scalar multiplication: Multiply all elements
- Matrix multiplication: (m×n) × (n×p) = (m×p)
- Hadamard product: Element-wise multiplication

Special Matrices:
- Diagonal: Non-zero only on diagonal
- Orthogonal: AᵀA = I
- Positive definite: xᵀAx > 0 for all x

Linear Transformations:
- Rotation, scaling, reflection
- Represented by matrices
- Composition = matrix multiplication

Eigenvalues and Eigenvectors:
- Av = λv (v is eigenvector, λ is eigenvalue)
- Principal directions of transformation
- Used in PCA, spectral clustering

Matrix Decomposition:
- LU: Lower and upper triangular
- QR: Orthogonal and upper triangular
- SVD: U Σ Vᵀ (most important for ML)
- Eigendecomposition: V Λ V⁻¹

Applications in ML:
- Data as matrix (samples × features)
- Weights in neural networks
- PCA uses eigendecomposition
- SVD for dimensionality reduction
- Covariance matrix analysis""",
        "source": "educational",
        "topic": "statistics"
    },
    {
        "title": "Calculus for Machine Learning",
        "content": """Calculus, particularly derivatives and gradients, is fundamental to understanding how machine learning models are optimized. Gradient descent, the primary optimization algorithm, relies heavily on calculus concepts.

Derivatives:
- Rate of change of a function
- f'(x) = lim[h→0] (f(x+h) - f(x)) / h
- Slope of tangent line at a point

Derivative Rules:
- Power rule: d/dx(xⁿ) = nxⁿ⁻¹
- Sum rule: d/dx(f+g) = f' + g'
- Product rule: d/dx(fg) = f'g + fg'
- Chain rule: d/dx(f(g(x))) = f'(g(x)) × g'(x)

Common Derivatives:
- d/dx(eˣ) = eˣ
- d/dx(ln x) = 1/x
- d/dx(sin x) = cos x
- d/dx(sigmoid(x)) = sigmoid(x)(1-sigmoid(x))

Partial Derivatives:
- Derivative with respect to one variable
- ∂f/∂x: Hold other variables constant
- Used for multivariable functions

Gradient:
- Vector of partial derivatives
- ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
- Points in direction of steepest increase
- Magnitude = rate of increase

Gradient Descent:
- Optimization algorithm
- Move in direction opposite to gradient
- θ = θ - α∇J(θ)
- α is learning rate

Chain Rule in Deep Learning:
- Backpropagation uses chain rule
- Compute gradients layer by layer
- ∂L/∂w = ∂L/∂y × ∂y/∂w

Hessian Matrix:
- Matrix of second derivatives
- Used in second-order optimization
- Describes curvature

Jacobian Matrix:
- Matrix of first-order partial derivatives
- For vector-valued functions
- Important in neural networks""",
        "source": "educational",
        "topic": "statistics"
    },

    # =========================================================================
    # DATA ENGINEERING (5 documents)
    # =========================================================================
    {
        "title": "Data Preprocessing Pipeline",
        "content": """Data preprocessing is the crucial first step in any machine learning project. Raw data is often messy, incomplete, and inconsistent. Proper preprocessing ensures your model receives clean, well-structured data.

Data Cleaning Steps:

1. Handling Missing Values:
   - Identify: df.isnull().sum()
   - Remove: df.dropna()
   - Impute:
     * Mean/median for numerical
     * Mode for categorical
     * Predictive imputation (KNN, regression)

2. Handling Outliers:
   - Detection:
     * Z-score: |z| > 3
     * IQR: Q1-1.5×IQR or Q3+1.5×IQR
     * Visualization (box plots)
   - Treatment:
     * Remove
     * Cap (Winsorization)
     * Transform (log)

3. Data Type Conversion:
   - Dates to datetime
   - Categories to categorical dtype
   - Numeric strings to numbers

4. Handling Duplicates:
   - df.duplicated()
   - df.drop_duplicates()

5. Fixing Inconsistencies:
   - Standardize text (lowercase, strip spaces)
   - Consistent formatting
   - Merge similar categories

Data Transformation:

1. Normalization (Min-Max Scaling):
   - Scale to [0, 1]
   - X' = (X - min) / (max - min)
   - Sensitive to outliers

2. Standardization (Z-score):
   - Mean = 0, Std = 1
   - X' = (X - μ) / σ
   - Better for algorithms assuming normal distribution

3. Log Transformation:
   - Reduce skewness
   - Handle multiplicative relationships

4. Box-Cox Transformation:
   - Power transformation
   - Makes data more normal-like

Encoding Categorical Variables:
- One-hot encoding
- Label encoding
- Target encoding
- Binary encoding""",
        "source": "educational",
        "topic": "data-engineering"
    },
    {
        "title": "Working with SQL Databases",
        "content": """SQL (Structured Query Language) is essential for data scientists to extract, manipulate, and analyze data stored in relational databases. Understanding SQL enables efficient data retrieval and preparation.

Basic SQL Commands:

SELECT - Retrieve data:
SELECT column1, column2 FROM table_name;
SELECT * FROM table_name;  -- All columns
SELECT DISTINCT column FROM table;  -- Unique values

WHERE - Filter rows:
SELECT * FROM table WHERE condition;
Operators: =, !=, <, >, <=, >=, BETWEEN, IN, LIKE, IS NULL

ORDER BY - Sort results:
SELECT * FROM table ORDER BY column ASC/DESC;

LIMIT - Restrict rows:
SELECT * FROM table LIMIT 10;

Aggregate Functions:
- COUNT(*): Number of rows
- SUM(column): Total
- AVG(column): Average
- MIN/MAX(column): Minimum/Maximum

GROUP BY - Aggregate by groups:
SELECT category, COUNT(*) FROM table GROUP BY category;

HAVING - Filter groups:
SELECT category, AVG(price) FROM products
GROUP BY category HAVING AVG(price) > 100;

JOINs - Combine tables:
- INNER JOIN: Matching rows only
- LEFT JOIN: All from left + matching right
- RIGHT JOIN: All from right + matching left
- FULL OUTER JOIN: All rows from both

Subqueries:
SELECT * FROM table WHERE column IN (SELECT column FROM other_table);

Common Table Expressions (CTE):
WITH cte_name AS (
    SELECT ...
)
SELECT * FROM cte_name;

Window Functions:
- ROW_NUMBER(), RANK(), DENSE_RANK()
- LEAD(), LAG()
- SUM() OVER (PARTITION BY ...)

Python + SQL:
- sqlite3: Built-in library
- SQLAlchemy: ORM and engine
- pandas.read_sql(): Direct to DataFrame""",
        "source": "educational",
        "topic": "data-engineering"
    },
    {
        "title": "Big Data and Distributed Computing",
        "content": """Big Data refers to datasets that are too large or complex to be processed with traditional data processing tools. Distributed computing enables processing of such massive datasets across multiple machines.

Big Data Characteristics (5 Vs):
1. Volume: Massive amounts of data
2. Velocity: Speed of data generation
3. Variety: Different types (structured, unstructured)
4. Veracity: Data quality and accuracy
5. Value: Extracting insights from data

Apache Hadoop Ecosystem:

1. HDFS (Hadoop Distributed File System):
   - Distributed storage
   - Data replicated across nodes
   - Fault-tolerant

2. MapReduce:
   - Processing paradigm
   - Map: Transform data in parallel
   - Reduce: Aggregate results
   - Batch processing

3. YARN:
   - Resource management
   - Job scheduling

Apache Spark:
- 100x faster than MapReduce (in-memory)
- Unified analytics engine
- Components:
  * Spark Core: Basic functionality
  * Spark SQL: Structured data
  * MLlib: Machine learning
  * Spark Streaming: Real-time
  * GraphX: Graph processing

Spark Concepts:
- RDD: Resilient Distributed Dataset
- DataFrame: Distributed collection with schema
- Transformations: Lazy operations (map, filter)
- Actions: Trigger computation (collect, count)

PySpark Example:
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("example").getOrCreate()
df = spark.read.csv("data.csv", header=True)
df.filter(df.age > 30).groupBy("city").count().show()

Other Big Data Tools:
- Kafka: Message streaming
- Hive: SQL-like queries on Hadoop
- Presto: Distributed SQL engine
- Dask: Parallel computing in Python""",
        "source": "educational",
        "topic": "data-engineering"
    },
    {
        "title": "Data Pipeline and ETL",
        "content": """Data pipelines are automated systems that move and transform data from source systems to destinations for analysis. ETL (Extract, Transform, Load) is the traditional approach to building data pipelines.

ETL Process:

1. Extract:
   - Pull data from sources
   - Sources: databases, APIs, files, streams
   - Handle different formats
   - Incremental vs full extraction

2. Transform:
   - Clean and validate data
   - Apply business rules
   - Aggregate and join
   - Format conversions
   - Data enrichment

3. Load:
   - Store in destination
   - Data warehouses (Snowflake, BigQuery)
   - Data lakes (S3, Azure Data Lake)
   - Databases

ELT (Extract, Load, Transform):
- Modern approach for cloud data warehouses
- Load raw data first, transform in destination
- Leverages powerful cloud compute

Pipeline Orchestration Tools:

1. Apache Airflow:
   - Python-based
   - DAGs (Directed Acyclic Graphs)
   - Scheduling and monitoring
   - Most popular choice

2. Prefect:
   - Modern alternative to Airflow
   - Easier to use
   - Better error handling

3. Luigi (Spotify):
   - Pipeline dependency management
   - Built-in visualization

4. dbt (data build tool):
   - Transform data in warehouse
   - SQL-based
   - Version control for transforms

Pipeline Best Practices:
- Idempotency: Safe to rerun
- Logging and monitoring
- Error handling and retries
- Data validation
- Documentation
- Testing

Data Quality Checks:
- Schema validation
- Null checks
- Range validation
- Referential integrity
- Freshness monitoring

CI/CD for Data Pipelines:
- Version control (Git)
- Automated testing
- Staged deployments
- Rollback capability""",
        "source": "educational",
        "topic": "data-engineering"
    },
    {
        "title": "Cloud Platforms for Data Science",
        "content": """Cloud platforms provide scalable infrastructure and managed services for data science and machine learning. The three major providers each offer comprehensive ML/AI services.

Amazon Web Services (AWS):
Storage:
- S3: Object storage for data lakes
- RDS: Managed relational databases
- DynamoDB: NoSQL database

Compute:
- EC2: Virtual machines
- Lambda: Serverless functions
- EMR: Managed Spark/Hadoop

ML Services:
- SageMaker: End-to-end ML platform
- Comprehend: NLP
- Rekognition: Computer vision

Google Cloud Platform (GCP):
Storage:
- Cloud Storage: Object storage
- BigQuery: Serverless data warehouse
- Cloud SQL: Managed databases

Compute:
- Compute Engine: VMs
- Cloud Functions: Serverless
- Dataproc: Managed Spark

ML Services:
- Vertex AI: Unified ML platform
- AutoML: Automated model training
- Cloud Vision, Natural Language APIs

Microsoft Azure:
Storage:
- Blob Storage: Object storage
- Azure SQL: Managed databases
- Cosmos DB: Multi-model database

Compute:
- Virtual Machines
- Azure Functions: Serverless
- HDInsight: Managed Hadoop/Spark

ML Services:
- Azure ML: End-to-end platform
- Cognitive Services: Pre-built AI
- Azure Databricks: Unified analytics

Choosing a Cloud:
- AWS: Largest, most mature
- GCP: Best for big data, competitive pricing
- Azure: Best for Microsoft shops

Cost Optimization:
- Spot/Preemptible instances
- Reserved instances for steady workloads
- Auto-scaling
- Shut down unused resources
- Right-size instances

MLOps on Cloud:
- Model versioning
- Experiment tracking
- Model deployment
- Monitoring and logging
- CI/CD pipelines""",
        "source": "educational",
        "topic": "data-engineering"
    },
]

# URLs for web scraping (if needed)
ML_TOPICS_URLS = {
    "machine-learning": [
        "https://www.geeksforgeeks.org/machine-learning/",
        "https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/"
    ],
    "deep-learning": [
        "https://www.geeksforgeeks.org/introduction-deep-learning/",
        "https://www.analyticsvidhya.com/blog/2018/10/introduction-neural-networks-deep-learning/"
    ],
    "nlp": [
        "https://www.geeksforgeeks.org/natural-language-processing-overview/",
        "https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/"
    ]
}


async def seed_sample_documents():
    """Seed knowledge base with sample documents"""
    try:
        logger.info("Seeding sample documents...")
        print(f"📚 Preparing to add {len(SAMPLE_DOCUMENTS)} educational documents...")

        # Initialize components
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()

        # Process sample documents
        processed = doc_processor.process_batch(SAMPLE_DOCUMENTS)

        if processed:
            # Add to vector store
            success = vector_store.add_documents(processed)

            if success:
                logger.info(f"Successfully added {len(processed)} sample documents")

                # Print topic summary
                topics = {}
                for doc in SAMPLE_DOCUMENTS:
                    topic = doc.get('topic', 'general')
                    topics[topic] = topics.get(topic, 0) + 1

                print("\n📊 Documents added by topic:")
                for topic, count in sorted(topics.items()):
                    print(f"   - {topic}: {count} documents")

                return len(processed)
            else:
                logger.error("Failed to add documents to vector store")
                return 0
        else:
            logger.error("No documents were processed")
            return 0

    except Exception as e:
        logger.error(f"Error seeding sample documents: {e}")
        return 0


async def fetch_web_content(urls: List[str]) -> List[Dict]:
    """Fetch content from web URLs"""
    try:
        logger.info(f"Fetching content from {len(urls)} URLs...")

        scraper = ScraperAgent()
        results = await scraper.scrape_multiple(urls)

        documents = []
        for result in results:
            if result.get('success') and result.get('content'):
                documents.append({
                    'title': result.get('title', 'Web Article'),
                    'content': result['content'],
                    'source': 'web',
                    'url': result.get('url', ''),
                    'topic': 'ml-web-content'
                })

        logger.info(f"Successfully fetched {len(documents)} documents")
        return documents

    except Exception as e:
        logger.error(f"Error fetching web content: {e}")
        return []


async def show_kb_stats():
    """Show current knowledge base statistics"""
    try:
        vector_store = VectorStore()
        stats = vector_store.get_collection_stats()
        print("\n📈 Knowledge Base Statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Collection: {stats.get('collection_name', 'N/A')}")
    except Exception as e:
        print(f"   Error getting stats: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Seed EduNotes Knowledge Base")
    parser.add_argument("--sample", action="store_true", help="Add sample educational documents (30+)")
    parser.add_argument("--web", action="store_true", help="Fetch content from web")
    parser.add_argument("--topics", nargs="+", help="Topics to fetch (machine-learning, deep-learning, nlp)")
    parser.add_argument("--stats", action="store_true", help="Show KB statistics")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("   EduNotes Knowledge Base Seeding Tool")
    print("="*60)

    total_added = 0

    if args.stats:
        await show_kb_stats()

    if args.sample:
        count = await seed_sample_documents()
        total_added += count
        print(f"\n✅ Added {count} sample documents")

    if args.web and args.topics:
        # Collect URLs for specified topics
        urls = []
        for topic in args.topics:
            if topic in ML_TOPICS_URLS:
                urls.extend(ML_TOPICS_URLS[topic])

        if urls:
            # Fetch web content
            documents = await fetch_web_content(urls)

            if documents:
                # Process and add to KB
                doc_processor = DocumentProcessor()
                vector_store = VectorStore()

                processed = doc_processor.process_batch(documents)
                if processed:
                    vector_store.add_documents(processed)
                    total_added += len(processed)
                    print(f"✅ Added {len(processed)} web documents")

    if total_added > 0:
        print(f"\n🎉 Successfully seeded knowledge base with {total_added} documents!")
        await show_kb_stats()
    elif not args.stats:
        print("\n📖 Usage:")
        print("  python seed_data.py --sample              # Add 30+ educational documents")
        print("  python seed_data.py --stats               # Show KB statistics")
        print("  python seed_data.py --web --topics ml nlp # Fetch from web")
        print("\nExample:")
        print("  python scripts/seed_data.py --sample")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
