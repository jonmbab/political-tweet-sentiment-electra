# NLP-Based Interactive Decision Aids: Evaluating LLM Predictions on Political Tweets[cite: 3]

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)[cite: 3]
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-orange.svg)](https://colab.research.google.com/)[cite: 3]
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/)[cite: 3]
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-green.svg)](https://pandas.pydata.org/)[cite: 3]
[![Status](https://img.shields.io/badge/Status-Master%20Thesis%20Showcase-lightgrey.svg)]()[cite: 3]

> **Portfolio Showcase:** This repository originates from my Master’s thesis in Business Administration (specializing in Marketing & Customer Insight) at the University of Hamburg. It demonstrates a complete end-to-end data analytics pipeline—ranging from web scraping unstructured social media data and advanced NLP preprocessing to model inference testing and critical performance evaluation.[cite: 3]

---

## 🚀 Interactive Notebooks (Google Colab)

Um den GitHub-Viewer-Fehler bei Jupyter Notebooks zu umgehen, kannst du die Notebooks hier direkt per Klick in Google Colab öffnen und ausführen:

* **1. Data Scraping, Cleaning & Labeling:**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonmbab/political-tweet-sentiment-electra/blob/main/notebooks/01_scraping_cleaning_and_labeling.ipynb)

* **2. Descriptive Statistics & Sentiment Analysis:**  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jonmbab/political-tweet-sentiment-electra/blob/main/notebooks/02_descriptive_stats_and_analysis.ipynb)

---

## 🎯 Background & Marketing Context (The "Why")[cite: 3]

Traditional decision aids in marketing (such as recommendation systems or attribute matrices) typically rely on structured product data and rigid questionnaires[cite: 3]. 

* **The Research Core:** Can modern Natural Language Processing (NLP) and Large Language Models be leveraged to create interactive decision aids where users express their preferences freely in their own words?[cite: 3]
* **The Marketing Transfer:** Political communication represents one of the most complex choice environments[cite: 3]. Demonstrating how to analyze and structure unstructured political texts provides a methodological framework that translates directly to complex consumer decision-making in business (e.g., high-involvement financial services, automotive choices, or brand perception)[cite: 3].

---

## 🔄 Data Pipeline & Methodology[cite: 3]

This project covers the full data analytics lifecycle[cite: 3]:

1. **Data Collection & Scraping:**[cite: 3]
   * Automated extraction of **30,000 raw tweets** from **421 members of the German Bundestag (MdB)** across the 6 major political parties (CDU/CSU, SPD, Grüne, FDP, Die Linke, AfD) over a 63-day period using `snscrape` and `selenium`[cite: 3].
   * Automated scraping and regex mapping of politician profile bios to verify party affiliations[cite: 3].

2. **Advanced NLP Preprocessing & Feature Engineering:**[cite: 3]
   * **Hashtag Segmentation (`hashformers`):** Resolved compound social media hashtags (e.g., `#klimaschutzgesetz` $\rightarrow$ individual words) using GPT-2/BERT rerankers to retain thematic context without information loss[cite: 3]. *Note: As the original library lacked native TensorFlow support at the time, I successfully collaborated with the package maintainer to establish a dedicated TensorFlow branch for this project.*[cite: 3]
   * **Text Cleaning:** Filtered out non-German tweets, duplicates, URLs, emojis, and short tweets ($<7$ words), reducing the dataset to **20,530 clean tweets**[cite: 3]. Explicitly retained stop words to preserve crucial negation structures for sentiment and polarity[cite: 3].

3. **Model Inference & Critical Evaluation:**[cite: 3]
   * Utilized pre-trained German Transformer architectures (`ELECTRA` / `BERT`) via Hugging Face[cite: 3].
   * Transformed the classification task using single-label prefix formatting (multiplying the dataset to 123,180 instances)[cite: 3].
   * Conducted rigorous model evaluation using confusion matrices, classification reports, and **prediction score distributions** (analyzing model confidence via boxplots and histograms)[cite: 3].

---

## 📊 Key Findings & Analytical Insights[cite: 3]

* **The Accuracy Paradox:** While the model achieved an overall accuracy of **28%** on hold-out data, deeper inspection revealed an average model confidence score of **~91%** across predictions[cite: 3].
* **Content Analysis & Root Cause:** Wordclouds and descriptive statistics proved that the low classification accuracy was driven by the **nature of the data itself** rather than model failure: politician tweets frequently contained general, everyday conversation and lacked sharp ideological differentiation between parties[cite: 3].
* **Marketing Implications:** The descriptive analysis revealed that politicians underutilized Twitter as a targeted communication and customer (voter) acquisition tool, often failing to establish clear brand differentiation[cite: 3].

---

## 🚀 Key Takeaways for Practice[cite: 3]

This showcase highlights my ability to[cite: 3]:
- Handle noisy, large-scale, unstructured web data and transform it into clean, analytics-ready structures[cite: 3].
- Execute advanced text preprocessing techniques (regex, custom hashtag segmentation, language detection)[cite: 3].
- Go beyond surface-level metrics: critically investigate model behavior, confidence scores, and data limitations rather than treating ML outputs as black boxes[cite: 3].
- Bridge academic research with practical marketing analytics and strategic business insights[cite: 3].

---

## 🛠 Tech Stack[cite: 3]

* **Environment:** Google Colab (GPU)[cite: 3]
* **Programming Language:** Python[cite: 3]
* **Data Manipulation & Analysis:** Pandas, NumPy, Scikit-Learn[cite: 3]
* **NLP & Model Testing:** Hugging Face Transformers, PyTorch/TensorFlow, Hashformers[cite: 3]
* **Data Extraction & Web Scraping:** Selenium, BeautifulSoup, Snscrape[cite: 3]
* **Version Control:** Git & GitHub[cite: 3]

---

## 📂 Repository Structure[cite: 3]

```text
├── data/                  # Anonymized sample input schema and data structure[cite: 3]
├── notebooks/             # Google Colab notebooks (Scraping, Preprocessing, Inference & Evaluation)[cite: 3]
├── src/                   # Reusable Python scripts for scraping, cleaning & hashtag segmentation[cite: 3]
└── README.md              # Project documentation[cite: 3]