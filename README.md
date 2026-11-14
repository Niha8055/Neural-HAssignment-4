
# CS5720 – Home Assignment 4

**Student Name:** Niharika Mannepalli  
**Student ID:** 700#775394  
**Course:** CS5720 – Neural Network and Deep Learning  
**University:** University of Central Missouri  
**Semester:** Fall 2026

## Assignment Overview
This repository contains three Python programs demonstrating NLP preprocessing, named entity recognition, and scaled dot-product attention.

## Q1: NLP Preprocessing Pipeline
**File:** `nlp_preprocessing.py`

- Tokenizes a sentence using NLTK.
- Removes common English stopwords.
- Applies Porter stemming.
- Prints original, filtered, and stemmed tokens.

## Q2: Named Entity Recognition
**File:** `named_entity_recognition.py`

- Uses spaCy's `en_core_web_sm` English model.
- Detects named entities in the assignment sentence.
- Prints entity text, category, and start/end character positions.

## Q3: Scaled Dot-Product Attention
**File:** `scaled_dot_product_attention.py`

- Computes attention scores using Q and K transpose.
- Scales scores by the square root of the key dimension.
- Applies softmax to obtain attention weights.
- Multiplies attention weights by V to generate the final output.

## Installation
```bash
python3 -m pip install nltk spacy numpy
python3 -m spacy download en_core_web_sm
```

## How to Run
```bash
python3 nlp_preprocessing.py
python3 named_entity_recognition.py
python3 scaled_dot_product_attention.py
```

## Expected Output for Q3
```text
Attention Weights:
[[0.731 0.269]
 [0.269 0.731]]

Final Output:
[[2.076 3.076 4.076 5.076]
 [3.924 4.924 5.924 6.924]]
```

## Conclusion
This assignment demonstrates essential NLP techniques and the attention mechanism used in Transformer models.
  
