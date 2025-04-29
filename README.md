# Fake-News-Detection---Semantic-Classification

<div style="text-align: center; padding: 15px; border-radius: 5px; font-size:16px;">
    <p><b>By:</b></p>
    <p>Deepak TM</p>
    <p>Umesh Goyal</p>
</div>

## Project Overview

This project aims to build a robust system for detecting fake news using semantic classification techniques. In an era of pervasive digital information, the rapid spread of misinformation poses a significant threat to public discourse and trust. By leveraging Natural Language Processing (NLP), specifically the Word2Vec embedding method, this project develops supervised machine learning models capable of distinguishing between true and fake news articles based on the semantic meaning and patterns embedded within the text. The goal is to contribute to mitigating the impact of fake news by providing an automated classification tool.

## Problem Statement

The proliferation of fake news has become a critical global issue. With vast quantities of news content generated daily, differentiating credible information from deliberately misleading reports is increasingly challenging for individuals and societies. This project addresses the need for an automated and efficient method to classify news articles as true or fake, thereby helping to curb the spread of misinformation and uphold the integrity of information dissemination.

## Features and Objectives

The key features and objectives of this project include:

*   **Data Preparation:** Loading, combining, and preparing datasets of true and fake news.
*   **Text Preprocessing:** Implementing standard NLP cleaning techniques including lowercasing, punctuation removal, filtering irrelevant elements, and advanced techniques like POS tagging and lemmatization to focus on semantically relevant terms (nouns).
*   **Data Splitting:** Dividing the processed data into training and validation sets for model development and evaluation.
*   **Exploratory Data Analysis (EDA):** Analyzing characteristics of the processed text data, including text length distributions, frequent words (Word Clouds), and common multi-word phrases (N-grams) for both true and fake news categories to gain insights into their linguistic differences.
*   **Feature Extraction:** Converting text data into numerical vectors using the Word2Vec model to capture semantic relationships between words.
*   **Model Training and Evaluation:** Training and evaluating multiple supervised classification models (Logistic Regression, Decision Tree, Random Forest) on the vectorized data to determine their effectiveness in classifying news articles.

## Methodology

The project follows a standard machine learning pipeline for text classification:

### Data Processing & Preprocessing

1.  **Loading Data:** True and Fake news datasets (`True.csv`, `Fake.csv`) were loaded.
2.  **Labeling:** A `news_label` column was added (1 for True, 0 for Fake).
3.  **Merging:** The two datasets were combined into a single DataFrame.
4.  **Handling Nulls:** Rows with missing values in critical columns (`title`, `text`, `date`) were identified and dropped.
5.  **Combining Text:** The `title` and `text` columns were concatenated into a new `news_text` column, as both contribute to the overall content.
6.  **Text Cleaning:** The `news_text` was cleaned by:
    *   Converting text to lowercase.
    *   Removing text enclosed in square brackets.
    *   Removing punctuation.
    *   Removing words containing numbers.
7.  **Advanced Preprocessing (Semantic Focus):** The cleaned text underwent:
    *   **POS Tagging:** Identifying the grammatical role of each word.
    *   **Lemmatization:** Reducing words to their base or root form (e.g., "running" -> "run").
    *   **Filtering:** Keeping only words tagged as Nouns (NN, NNS) and removing standard English stopwords, aiming to retain key concepts and entities relevant to the news content.

### Tools Used

*   **Python:** The primary programming language.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **NLTK (Natural Language Toolkit):** For basic text processing tasks (though spaCy was primarily used for POS/Lemmatization).
*   **SpaCy:** For efficient POS tagging and lemmatization.
*   **Scikit-learn:** For data splitting, model training (Logistic Regression, Decision Tree, Random Forest), and evaluation metrics.
*   **Gensim:** For loading and using the pre-trained Word2Vec model.
*   **Matplotlib & Seaborn:** For data visualization (histograms, bar plots).
*   **WordCloud:** For generating word cloud visualizations.

### Models or Techniques Applied

*   **Word Embeddings (Word2Vec):** A pre-trained Word2Vec model (`word2vec-google-news-300`) was used to convert processed text into dense vector representations, capturing semantic meaning based on word co-occurrences. Document vectors were created by averaging the vectors of the words within each document.
*   **Supervised Classification Models:**
    *   Logistic Regression
    *   Decision Tree
    *   Random Forest

### Visualizations & Analysis

Exploratory Data Analysis was conducted on the training data to understand characteristics after preprocessing:

*   **Character Length Histograms:** Visualized the distribution of text lengths before and after lemma/POS filtering, showing the significant reduction in text size by keeping only nouns.
*   **Word Clouds:** Generated word clouds for true and fake news separately, highlighting the most frequent words in each category visually.
*   **N-gram Analysis:** Identified and displayed the top 10 unigrams, bigrams, and trigrams for true and fake news, revealing common single words and phrases characteristic of each class. This analysis showed distinct patterns, such as Reuters mentions being prominent in true news and specific sensational phrases or names appearing more frequently in fake news.

## Dependencies

The following Python libraries are required to run this project:

*   `numpy`
*   `pandas`
*   `nltk`
*   `spacy`
*   `scipy`
*   `pydantic` (Might be a dependency of other libs)
*   `wordcloud`
*   `scikit-learn`
*   `gensim`
*   `matplotlib`
*   `seaborn`

You also need to download the SpaCy English model and the Word2Vec model:

```bash
python -m spacy download en_core_web_sm
python -m gensim.downloader install word2vec-google-news-300
```

## Key Insights and Results

*   **Preprocessing Impact:** Filtering text to include only nouns (NN/NNS) significantly reduced the text size, focusing the feature extraction on core concepts and entities, potentially improving semantic capture by Word2Vec.
*   **Linguistic Differences:** EDA revealed distinct sets of frequent words and phrases in true versus fake news after preprocessing. True news often contained terms related to official sources ("reuters", "statement") and political processes ("government", "state"), while fake news frequently featured sensational terms, specific names often associated with conspiracy theories, and phrases related to online dissemination ("century wire", "image").
*   **Model Performance:**
    *   Logistic Regression achieved an accuracy of ~0.901 and an F1-score of ~0.896.
    *   Decision Tree achieved lower performance with an accuracy of ~0.823 and an F1-score of ~0.812.
    *   Random Forest achieved the highest accuracy of ~0.901 and a slightly higher F1-score of ~0.895 compared to Logistic Regression (depending on rounding, they are very close, but RF generally shows marginally better balanced scores).

The models trained on Word2Vec vectors performed reasonably well, particularly Logistic Regression and Random Forest, demonstrating the ability of semantic embeddings to capture patterns indicative of fake news. The superior performance of RF and LR compared to DT suggests that the relationship between semantic features and news authenticity might be better captured by linear models or ensemble methods in this context.

## Conclusion

This project successfully implemented a fake news detection system using semantic classification with Word2Vec embeddings and supervised learning models. By focusing on nouns and utilizing pre-trained word vectors, the models were able to achieve promising results. The Exploratory Data Analysis highlighted clear linguistic differences between true and fake news in the processed text, validating the semantic approach.

The Random Forest classifier emerged as the best-performing model based on the evaluation metrics (Accuracy and F1-score) on the validation set. While the results are encouraging, particularly the ability of the models to distinguish between classes based on semantic features, limitations include the reliance on a specific pre-trained Word2Vec model and the scope of the preprocessing steps.

## Actionable Outcomes or Recommendations

1.  **Model Selection:** The Random Forest model is recommended for deployment or further development based on its robust performance. Logistic Regression is a strong alternative due to its simplicity and comparable results.
2.  **Further Optimization:** Hyperparameter tuning for the Random Forest and Logistic Regression models could potentially improve performance further. Cross-validation should be used for more reliable performance estimates.
3.  **Explore Other Embeddings:** Investigate other advanced word embedding techniques (e.g., GloVe, FastText) or contextual embeddings (e.g., BERT, RoBERTa) to see if they capture more nuanced semantic features that improve classification.
4.  **Dataset Augmentation:** Incorporating more diverse and potentially more recent news data could enhance the model's generalization capabilities.
5.  **Feature Engineering:** Explore combining semantic features with other features like sentiment, writing style analysis, or metadata (though 'date' was dropped here, other forms of metadata might be relevant).

## Contributors

*   Deepak TM
*   Umesh Goyal

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. (Note: Create a LICENSE file if one doesn't exist in the repo).

## Contact Information

For any inquiries regarding the project, please contact the contributors via their GitHub profiles or through the repository issues section.
