"""
Script to seed the knowledge base with initial ML/AI content
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

# Sample ML/AI content for seeding
SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to Machine Learning",
        "content": """Machine Learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Machine learning algorithms are categorized as supervised, unsupervised, semi-supervised, and reinforcement learning. Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data.""",
        "source": "educational",
        "topic": "machine-learning"
    },
    {
        "title": "Neural Networks Fundamentals",
        "content": """Artificial Neural Networks (ANNs) are computing systems inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain.

Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals then processes them and can signal neurons connected to it. The signal at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs.

Neural networks learn by adjusting the weights of connections between neurons. This process, called training, involves using algorithms like backpropagation to minimize the difference between predicted and actual outputs. Deep learning refers to neural networks with multiple hidden layers between input and output layers.""",
        "source": "educational",
        "topic": "deep-learning"
    },
    {
        "title": "Support Vector Machines (SVM)",
        "content": """Support Vector Machines are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. Given a set of training examples, each marked as belonging to one of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other.

An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on which side of the gap they fall.

The key idea behind SVMs is to find the hyperplane that best separates the classes in the feature space. This hyperplane is chosen to maximize the margin between the classes, where the margin is defined as the distance between the hyperplane and the nearest data points from each class, called support vectors.""",
        "source": "educational",
        "topic": "machine-learning"
    },
    {
        "title": "Natural Language Processing Basics",
        "content": """Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.

Key NLP tasks include tokenization (breaking text into words or phrases), part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. Modern NLP heavily relies on machine learning, particularly deep learning models like transformers.

The transformer architecture, introduced in 2017, revolutionized NLP by enabling models to process entire sequences simultaneously rather than sequentially. This led to breakthrough models like BERT, GPT, and T5, which achieve state-of-the-art performance on various NLP tasks through pre-training on large text corpora followed by fine-tuning on specific tasks.""",
        "source": "educational",
        "topic": "nlp"
    },
    {
        "title": "Reinforcement Learning Concepts",
        "content": """Reinforcement Learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward. It differs from supervised learning in that labelled input/output pairs are not needed, and sub-optimal actions need not be explicitly corrected.

The basic RL framework consists of an agent, environment, states, actions, and rewards. The agent learns to achieve a goal in an uncertain, potentially complex environment. The agent uses trial and error to come up with a solution to the problem. To get the machine to do what the programmer wants, the artificial intelligence gets either rewards or penalties for the actions it performs.

Key algorithms in RL include Q-learning, Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic methods. Applications range from game playing (like AlphaGo) to robotics, autonomous vehicles, and recommendation systems.""",
        "source": "educational",
        "topic": "reinforcement-learning"
    }
]

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

async def main():
    parser = argparse.ArgumentParser(description="Seed EduNotes Knowledge Base")
    parser.add_argument("--sample", action="store_true", help="Add sample ML documents")
    parser.add_argument("--web", action="store_true", help="Fetch content from web")
    parser.add_argument("--topics", nargs="+", help="Topics to fetch (machine-learning, deep-learning, nlp)")
    
    args = parser.parse_args()
    
    total_added = 0
    
    if args.sample:
        count = await seed_sample_documents()
        total_added += count
        print(f"✅ Added {count} sample documents")
    
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
    else:
        print("\nUsage:")
        print("  python seed_data.py --sample")
        print("  python seed_data.py --web --topics machine-learning deep-learning")

if __name__ == "__main__":
    asyncio.run(main())