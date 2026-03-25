# Risk-Aware Dialogue Framework for Safer LLM-Powered Socially Assistive Robots

This repository contains the implementation and experimental results of a **risk-aware dialogue framework** designed to improve the safety of **LLM-powered socially assistive robots (SARs)** in elderly care.

The framework introduces:
- **Query Risk Assessment Module (QRAM)**
- **Query Risk Score (QRS)**
- **Risk-Aware Response Safety Score (RRSS)**

It supports two strategies:
1. **Risk-aware response refinement**
2. **Risk-aware response generation**

---

## 📂 Repository Structure

├── Data/   
│ └── Elderly_Query.xlsx   
│   
├── Implementation/    
│ ├── main.ipynb      
│ ├── Generated_raw_response.ipynb    
│ ├── Generate_refined_response.ipynb    
│ ├── Generated_regenerate_response.ipynb    
│ ├── module_raw_response.py    
│ ├── module_refined_response.py    
│ ├── module_generated_response.py    
│ ├── QRSCalculator.py     
│ ├── QRSDetection.py     
│ ├── rrss_calculator.py    
│ ├── run_rrss.ipynb    
│ ├── plot_rrss.ipynb    
│ └── Plot Results.ipynb     
│     
├── Results/    
│ ├── Figures/     
│ ├── QRS_results.csv    
│ ├── robot_generated_responses_model_.json    
│ ├── robot_raw_responses_model_.json    
│ ├── robot_refined_responses_model.json    
│ ├── rrss_final.xlsx     
│ └── rrss_summary.xlsx     
│     
└── README.md    


---

## 📊 Dataset

The dataset is located in the `Data/` folder:

- **Elderly_Query.xlsx**
  - Contains **230 user queries** simulating real-world interactions with elderly users.
  - Includes:
    - Daily assistance requests  
    - Information queries  
    - Health-related statements  
    - Emergency scenarios  
    - Monitoring/support requests  

---

## ⚙️ Implementation

The `Implementation/` folder contains all core components:

### 🔹 Response Generation
- `module_raw_response.py` → Raw LLM responses  
- `module_refined_response.py` → Risk-aware refinement  
- `module_generated_response.py` → Risk-aware generation  

### 🔹 Risk Assessment
- `QRSDetection.py` → Extracts risk indicators  
- `QRSCalculator.py` → Computes Query Risk Score (QRS)  

### 🔹 Evaluation
- `rrss_calculator.py` → Computes RRSS metric  

### 🔹 Notebooks
- `main.ipynb` → Main pipeline  
- `Generated_raw_response.ipynb` → Raw responses  
- `Generate_refined_response.ipynb` → Refinement  
- `Generated_regenerate_response.ipynb` → Regeneration  
- `run_rrss.ipynb` → RRSS evaluation  
- `plot_rrss.ipynb`, `Plot Results.ipynb` → Visualization  

---

## 📈 Results

The `Results/` folder includes:

- **QRS_results.csv** → Computed risk scores  
- **JSON files** → Model outputs:
  - Raw responses  
  - Refined responses  
  - Regenerated responses  
- **rrss_final.xlsx / rrss_summary.xlsx** → Evaluation results  
- **Figures/** → Plots used in the paper  

---

## 🧠 Models Used

The framework supports multiple LLMs:

- DeepSeek-R1  
- LLaMA 3.1  
- Phi-4 (14B)  
- Qwen  
- Mistral  

---

## 🚀 How to Run

1. Install dependencies
```bash
pip install -r requirements.txt


Jupyter Notebook main.ipynb

jupyter notebook run_rrss.ipynb
