# 🧠 TV Series Analysis System 

An end-to-end journey in **Natural Language Processing (NLP)**.  
I started from gathering raw data and went all the way to building a **character-based chatbot**.  
Along the way, I explored web scraping, neural networks, Transformers, entity recognition, network graphs, text classification, and conversational AI.  

---

## 📚 What I Did  

- **Gathered Data**: I scraped text data from the internet using **Scrapy**.  
- **Explored Neural Networks → LLMs**: I studied how neural networks evolved into today’s **large language models**.  
- **Built a Zero-Shot Classifier**: I used **Hugging Face Transformers** to classify text without training data.  
- **Extracted Named Entities**: With **SpaCy**, I extracted characters and entities from the text.  
- **Created a Character Network**: Using **SpaCy NER + NetworkX + PyViz**, I built a graph showing how characters relate to each other.  
- **Trained a Custom Text Classifier**: I fine-tuned a model on a dataset using Hugging Face to reach state-of-the-art results.  
- **Built a Chatbot**: Finally, I created a chatbot that imitates my favorite characters so I could have a conversation with them.  

---

## 🛢 Datasets I Used  

- Naruto Subtitles Dataset  
- Anime Text Dataset (Kaggle)  
- Naruto Fandom Wiki Data  

---

## 🖥️ Tools & Libraries  

- **Python**  
- **Scrapy** (web scraping)  
- **Hugging Face Transformers** (zero-shot classification, text classification, chatbot)  
- **SpaCy** (NER for entity extraction)  
- **NetworkX + PyViz** (character graph visualization)  
- **Torch** (deep learning backend)  

---

## 🚀 How I Ran It  

### 🔹 Locally  

```bash
git clone https://github.com/MennatallaHussein/nlp-project-clean.git
cd nlp-project-clean


### 🔹 On Google Colab (with GPU)  

1. I enabled **GPU** from `Runtime → Change runtime type`.  

2. I cloned my repo:  
```bash
!git clone https://github.com/MennatallaHussein/nlp-project-clean.git
%cd nlp-project-clean



nlp-project-clean/
│── data/              # Datasets I collected
│── scrapy/            # My web scraping spiders
│── models/            # Trained Hugging Face models
│── notebooks/         # Colab & Jupyter experiments
│── src/               # Core code (NER, classifiers, chatbot)
│── outputs/           # Results, graphs, trained models
│── requirements.txt   # Dependencies
│── main.py            # Example pipeline script
│── README.md          # This file



pip install -r requirements.txt
python main.py   # or the module I wanted to run
