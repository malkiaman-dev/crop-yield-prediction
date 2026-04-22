# Crop Yield Prediction System

## Overview

The Crop Yield Prediction System is a machine learning-based web application built with Flask that predicts crop yield based on environmental and agricultural inputs.

Users can enter:

* Year
* Average Rainfall (mm/year)
* Pesticides Used (tonnes)
* Average Temperature (°C)
* Area / Country
* Crop Item

The system processes the input using a trained machine learning model and generates:

* Predicted crop yield
* A downloadable structured PDF report

---

## Features

### 1. Smart Crop Yield Prediction

Uses a trained Decision Tree Regressor model (`dtr.pkl`) with a preprocessing pipeline (`preprocessor.pkl`) to predict yield.

### 2. Input Validation

Validates:

* Missing fields
* Invalid numeric values
* Out-of-range values
* Unknown country/crop names

### 3. Auto Suggestions

Uses dataset values from `yield_df.csv` to provide:

* Country suggestions
* Crop suggestions

### 4. Structured PDF Report

Generates a professional PDF report containing:

* Prediction summary
* Input table
* Environmental conditions
* Generated date/time
* Developer details

### 5. Premium UI

Responsive and modern interface with:

* Glassmorphism card design
* Responsive layout
* Error handling
* Clear fields option

---

## Tech Stack

### Backend

* Python
* Flask
* Pandas
* Scikit-learn
* ReportLab

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript

### Machine Learning

* Decision Tree Regressor
* Preprocessing Pipeline

---

## Project Structure

```bash
crop-yield/
│
├── app.py
├── dtr.pkl
├── preprocessor.pkl
├── yield_df.csv
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## Installation

### 1. Clone project

```bash
git clone <repository-url>
cd crop-yield
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install flask pandas scikit-learn reportlab
```

---

## Run Project

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000/
```

---

## Example Input

| Field             | Example  |
| ----------------- | -------- |
| Year              | 2013     |
| Average Rainfall  | 800      |
| Pesticides Tonnes | 120      |
| Avg Temperature   | 23       |
| Country           | Pakistan |
| Crop Item         | Wheat    |

---

## Output

The system will provide:

* Predicted crop yield
* Structured downloadable PDF report

---

## Developer

**Developed by:** Malki Aman

**Prepared by:** Crop Yield Prediction System

**Prediction Model:** Machine Learning-Based Analysis
