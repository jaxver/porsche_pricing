# Elferspot Listings - Porsche Price Analytics

A comprehensive Python application for analyzing Porsche car listings data with advanced machine learning capabilities for price prediction and market analysis. This project combines web scraping, data processing, and predictive modeling to provide insights into the Porsche vehicle market.

## 🚗 Features

- **Data Processing Pipeline**: Bronze → Silver → Gold data transformation workflow
- **Machine Learning Models**: CatBoost-based price prediction with cross-validation
- **Interactive Dashboard**: Streamlit web application for browsing and analyzing listings
- **Price Analytics**: Identify underpriced listings and market trends
- **Multi-Model Analysis**: Compare Ridge, ElasticNet, and CatBoost regression models
- **Currency Conversion**: Automatic EUR conversion with exchange rate management

## 📁 Project Structure

```
elferspot_prod/
├── app/
│   └── streamlit_app.py          # Interactive dashboard application
├── elferspot_listings/           # Main library package
│   ├── __init__.py
│   └── notebooks/                # Production notebooks
│       ├── bronze_to_silver_production.ipynb
│       └── listing_scores_production.ipynb
├── notebooks/                    # Research and development notebooks
│   ├── Catboost model.ipynb      # CatBoost model training
│   ├── Listings_bronzetosilver.ipynb  # Data transformation
│   ├── Predictive_regression.ipynb    # Price prediction analysis
│   ├── scrape_911.ipynb          # Web scraping utilities
│   └── ...                       # Additional analysis notebooks
├── scripts/
│   └── exchange_rates.py         # Currency conversion utilities
├── tests/
│   └── test_basic.py             # Unit tests
├── data/                         # Data files (excluded from git)
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Elferspot-Scraper.git
cd Elferspot-Scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data directories (data files not included in repository):
```bash
mkdir -p data results logs
```

### Running the Dashboard

Launch the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- Interactive filtering by model, year, mileage, price
- Time series visualizations
- Model-based analytics and comparisons
- Direct links to individual listings

## 📊 Data Pipeline

1. **Bronze Layer**: Raw scraped data from various sources
2. **Silver Layer**: Cleaned and standardized data with feature engineering
3. **Gold Layer**: Final analytical dataset with predictions and scores

## 🤖 Machine Learning Models

The project implements multiple regression approaches:

- **CatBoost**: Primary model with categorical feature handling
- **Ridge Regression**: Linear baseline with L2 regularization
- **ElasticNet**: Combined L1/L2 regularization approach

Models are evaluated using cross-validation with metrics including RMSE, MAE, and R².

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Jupyter Notebooks

The `notebooks/` directory contains research and analysis notebooks:
- Data exploration and visualization
- Feature engineering experiments
- Model training and evaluation
- Outlier detection and analysis

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from public Porsche listing platforms
- Built with Python, pandas, scikit-learn, CatBoost, and Streamlit

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Always respect website terms of service when scraping data.