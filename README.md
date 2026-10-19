# NLP-Based Interactive Decision Aids: Evaluating LLM Predictions on Political Tweets

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-orange.svg)](https://colab.research.google.com/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Master%20Thesis%20Showcase-lightgrey.svg)]()

> **Portfolio Showcase:** This repository originates from my Master’s thesis in Business Administration (specializing in Marketing & Customer Insight) at the University of Hamburg. It demonstrates a complete end-to-end data analytics pipeline—ranging from web scraping unstructured social media data and advanced NLP preprocessing to model inference testing and critical performance evaluation.

---

## 🎯 Background & Marketing Context (The "Why")

Traditional decision aids in marketing (such as recommendation systems or attribute matrices) typically rely on structured product data and rigid questionnaires. 

* **The Research Core:** Can modern Natural Language Processing (NLP) and Large Language Models be leveraged to create interactive decision aids where users express their preferences freely in their own words?
* **The Marketing Transfer:** Political communication represents one of the most complex choice environments. Demonstrating how to analyze and structure unstructured political texts provides a methodological framework that translates directly to complex consumer decision-making in business (e.g., high-involvement financial services, automotive choices, or brand perception).

---

## 🔄 Data Pipeline & Methodology

This project covers the full data analytics lifecycle:

1. **Data Collection & Scraping:**
   * Automated extraction of **30,000 raw tweets** from **421 members of the German Bundestag (MdB)** across the 6 major political parties (CDU/CSU, SPD, Grüne, FDP, Die Linke, AfD) over a 63-day period using `snscrape` and `selenium`.
   * Automated scraping and regex mapping of politician profile bios to verify party affiliations.

2. **Advanced NLP Preprocessing & Feature Engineering:**
   * **Hashtag Segmentation (`hashformers`):** Resolved compound social media hashtags (e.g., `#klimaschutzgesetz` $\rightarrow$ individual words) using GPT-2/BERT rerankers to retain thematic context without information loss. *Note: As the original library lacked native TensorFlow support at the time, I successfully collaborated with the package maintainer to establish a dedicated TensorFlow branch for this project.*
   * **Text Cleaning:** Filtered out non-German tweets, duplicates, URLs, emojis, and short tweets ($<7$ words), reducing the dataset to **20,530 clean tweets**. Explicitly retained stop words to preserve crucial negation structures for sentiment and polarity.

3. **Model Inference & Critical Evaluation:**
   * Utilized pre-trained German Transformer architectures (`ELECTRA` / `BERT`) via Hugging Face.
   * Transformed the classification task using single-label prefix formatting (multiplying the dataset to 123,180 instances).
   * Conducted rigorous model evaluation using confusion matrices, classification reports, and **prediction score distributions** (analyzing model confidence via boxplots and histograms).

---

## 📊 Key Findings & Analytical Insights

* **The Accuracy Paradox:** While the model achieved an overall accuracy of **28%** on hold-out data, deeper inspection revealed an average model confidence score of **~91%** across predictions.
* **Content Analysis & Root Cause:** Wordclouds and descriptive statistics proved that the low classification accuracy was driven by the **nature of the data itself** rather than model failure: politician tweets frequently contained general, everyday conversation and lacked sharp ideological differentiation between parties.
* **Marketing Implications:** The descriptive analysis revealed that politicians underutilized Twitter as a targeted communication and customer (voter) acquisition tool, often failing to establish clear brand differentiation.

---

## 🚀 Key Takeaways for Practice

This showcase highlights my ability to:
- Handle noisy, large-scale, unstructured web data and transform it into clean, analytics-ready structures.
- Execute advanced text preprocessing techniques (regex, custom hashtag segmentation, language detection).
- Go beyond surface-level metrics: critically investigate model behavior, confidence scores, and data limitations rather than treating ML outputs as black boxes.
- Bridge academic research with practical marketing analytics and strategic business insights.

---

## 🛠 Tech Stack

* **Environment:** Google Colab (GPU)
* **Programming Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy, Scikit-Learn
* **NLP & Model Testing:** Hugging Face Transformers, PyTorch/TensorFlow, Hashformers
* **Data Extraction & Web Scraping:** Selenium, BeautifulSoup, Snscrape
* **Version Control:** Git & GitHub

---

## 📂 Repository Structure

```text
├── data/                  # Anonymized sample input schema and data structure
├── notebooks/             # Google Colab notebooks (Scraping, Preprocessing, Inference & Evaluation)
├── src/                   # Reusable Python scripts for scraping, cleaning & hashtag segmentation
└── README.md              # Project documentation