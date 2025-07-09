# Medium Models (100K-1M Parameters)
*The sweet spot between simplicity and power*

**Created by Gemini CLI**

Medium-scale models that offer a good balance of computational efficiency and performance. These implementations are perfect for real-world applications where you need more than basic models but don't want to break the bank on computing resources.

## The Collection

### `kmeans_100k_param.py`
**Parameters:** 100,000  
**Purpose:** K-means clustering at scale

A robust clustering algorithm that can handle large datasets efficiently. Perfect for customer segmentation, data exploration, and unsupervised learning tasks.

### `random_forest_1m_param.py`
**Parameters:** 1 million  
**Purpose:** Ensemble learning with decision trees

A powerful ensemble method that combines multiple decision trees to create a robust predictor. Excellent for both classification and regression tasks with built-in feature importance.

## Performance Characteristics

These models offer:
- **Reasonable training times** on standard hardware
- **Good generalization** without overfitting
- **Interpretable results** (especially Random Forest)
- **Scalable architecture** for production use

## Use Cases

- **Customer segmentation** with K-means
- **Feature selection** and importance ranking
- **Baseline models** for comparison
- **Production deployment** where efficiency matters

## Training Requirements

- **RAM:** 4-8GB recommended
- **Training time:** Minutes to hours (depending on dataset)
- **Data size:** Thousands to millions of samples
- **Hardware:** CPU sufficient, GPU optional

---
*"Not too big, not too small, just right for getting things done"*