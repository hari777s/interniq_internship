# Task 1: Prediction of House Prices

## Project Overview
This project focuses on predicting house prices using machine learning techniques based on a dataset of property features. The goal is to build a predictive model that accurately forecasts house prices based on factors like location, size, and other attributes.

## Dataset
The dataset used includes various features such as:
- **Location**: Geographic information.
- **Size**: Number of rooms, area in square feet.
- **Age of Property**: Year built or last renovated.
- **Amenities**: Additional features like parking, garden, etc.

## Project Structure
- **data/**: Contains the dataset used for training and testing.
- **notebooks/**: Jupyter notebooks for exploratory data analysis (EDA) and model building.
- **models/**: Saved models and serialized objects.
- **scripts/**: Python scripts for data processing, training, and evaluation.

## Key Files
- `eda.ipynb`: Initial data exploration and visualization.
- `train_model.py`: Script to preprocess data and train the model.
- `predict.py`: Script to load the trained model and predict house prices.

## Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib

To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preprocessing**: Clean and preprocess data using `data_preprocessing.py`.
2. **Training**: Train the model using `train_model.py`.
3. **Prediction**: Use `predict.py` to predict prices on new data.

## Results
Include evaluation metrics such as Mean Absolute Error (MAE) and R-squared scores to assess model performance.

## Contributing
Feel free to contribute by opening issues or pull requests for improvements.
