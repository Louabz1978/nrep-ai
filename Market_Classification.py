import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class HousingMarketPredictor:
    """
    Complete XGBoost model for housing market condition prediction
    Classifies markets as: 0=Buyers Market, 1=Even Market, 2=Sellers Market
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False

    def create_sample_data(self, n_samples=1000):
        """
        Generate realistic sample housing market data
        """
        np.random.seed(42)

        # Generate base metrics with realistic correlations
        data = {}

        # Market conditions (0=Buyers, 1=Even, 2=Sellers)
        market_conditions = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])

        # Generate features based on market conditions
        for i in range(n_samples):
            condition = market_conditions[i]

            if condition == 0:  # Buyers Market
                # High inventory, low absorption, longer DOM
                active_listings = np.random.normal(500, 100)
                closed_sales = np.random.normal(80, 20)
                days_on_market = np.random.normal(45, 15)
                pending_conversion = np.random.normal(0.75, 0.1)
                back_to_market_rate = np.random.normal(0.15, 0.05)

            elif condition == 1:  # Even Market
                # Balanced metrics
                active_listings = np.random.normal(300, 50)
                closed_sales = np.random.normal(100, 15)
                days_on_market = np.random.normal(30, 10)
                pending_conversion = np.random.normal(0.85, 0.05)
                back_to_market_rate = np.random.normal(0.08, 0.03)

            else:  # Sellers Market
                # Low inventory, high absorption, quick sales
                active_listings = np.random.normal(150, 30)
                closed_sales = np.random.normal(120, 25)
                days_on_market = np.random.normal(15, 8)
                pending_conversion = np.random.normal(0.95, 0.03)
                back_to_market_rate = np.random.normal(0.03, 0.02)

            # Ensure positive values
            active_listings = max(50, active_listings)
            closed_sales = max(10, closed_sales)
            days_on_market = max(5, days_on_market)
            pending_conversion = np.clip(pending_conversion, 0.5, 1.0)
            back_to_market_rate = np.clip(back_to_market_rate, 0.01, 0.3)

            # Calculate derived features
            total_properties = active_listings + closed_sales + np.random.normal(50, 20)
            total_properties = max(100, total_properties)

            pending_listings = np.random.normal(closed_sales * 0.3, 10)
            pending_listings = max(5, pending_listings)

            new_listings = np.random.normal(closed_sales * 1.2, 20)
            new_listings = max(20, new_listings)

            off_market = np.random.normal(active_listings * 0.1, 5)
            off_market = max(1, off_market)

            data[i] = {
                'active_listings': active_listings,
                'closed_sales': closed_sales,
                'pending_listings': pending_listings,
                'new_listings': new_listings,
                'off_market': off_market,
                'total_properties': total_properties,
                'days_on_market': days_on_market,
                'pending_conversion_rate': pending_conversion,
                'back_to_market_rate': back_to_market_rate,
                'market_condition': condition
            }

        return pd.DataFrame.from_dict(data, orient='index')

    def engineer_features(self, df):
        """
        Create all the features we discussed for market prediction
        """
        features_df = df.copy()

        # Activity Ratios
        features_df['new_listings_ratio'] = features_df['new_listings'] / features_df['total_properties']
        features_df['absorption_rate'] = features_df['closed_sales'] / features_df['active_listings']
        features_df['off_market_rate'] = features_df['off_market'] / features_df['active_listings']

        # Velocity Metrics
        features_df['market_turnover'] = (features_df['closed_sales'] + features_df['pending_listings']) / features_df['total_properties']
        features_df['pending_duration'] = features_df['days_on_market'] * 0.3  # Estimated pending time

        # Supply-Demand Indicators
        features_df['inventory_months'] = features_df['active_listings'] / (features_df['closed_sales'] / 3)
        features_df['competition_index'] = features_df['pending_listings'] / features_df['active_listings']
        features_df['market_pressure'] = (features_df['new_listings'] - features_df['closed_sales']) / features_df['new_listings']

        # Market Heat Score (composite metric)
        features_df['market_heat_score'] = (
            (1 / features_df['inventory_months']) * 0.3 +
            features_df['absorption_rate'] * 0.25 +
            (1 / features_df['days_on_market']) * 0.2 +
            features_df['pending_conversion_rate'] * 0.15 +
            (1 - features_df['back_to_market_rate']) * 0.1
        )

        # Price Momentum Indicators (simulated)
        features_df['price_momentum'] = np.where(
            features_df['market_heat_score'] > features_df['market_heat_score'].median(),
            np.random.normal(1.05, 0.1, len(features_df)),  # Rising prices
            np.random.normal(0.98, 0.08, len(features_df))  # Falling prices
        )

        # Market Efficiency Metrics
        features_df['listing_efficiency'] = features_df['closed_sales'] / features_df['new_listings']
        features_df['market_velocity'] = features_df['total_properties'] / features_df['days_on_market']

        # Handle infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())

        return features_df

    def prepare_features(self, df):
        """
        Select and prepare final feature set for modeling
        """
        feature_columns = [
            'new_listings_ratio', 'absorption_rate', 'off_market_rate',
            'market_turnover', 'pending_duration', 'inventory_months',
            'competition_index', 'market_pressure', 'market_heat_score',
            'price_momentum', 'listing_efficiency', 'market_velocity',
            'days_on_market', 'pending_conversion_rate', 'back_to_market_rate'
        ]

        self.feature_names = feature_columns
        return df[feature_columns]

    def fit(self, X, y, optimize_hyperparameters=True):
        """
        Train the XGBoost model with optional hyperparameter optimization
        """
        print("🚀 Training XGBoost Housing Market Prediction Model...")

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        if optimize_hyperparameters:
            print("🔧 Optimizing hyperparameters...")

            # Define parameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            }

            # Create XGBoost classifier
            xgb_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1
            )

            # Perform grid search
            grid_search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_

            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

        else:
            # Use default optimized parameters
            self.model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1
            )

            self.model.fit(X_train_scaled, y_train)

        # Validate model
        val_predictions = self.model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, val_predictions)

        print(f"✅ Validation Accuracy: {val_accuracy:.4f}")

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"✅ Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions on new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, plot=True):
        """
        Get and visualize feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        if plot:
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance - XGBoost Housing Market Model')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.show()

        return importance_df

    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        print("📊 Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                  target_names=['Buyers Market', 'Even Market', 'Sellers Market']))

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Buyers', 'Even', 'Sellers'],
                   yticklabels=['Buyers', 'Even', 'Sellers'])
        plt.title('Confusion Matrix - Housing Market Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def interpret_prediction(self, X_sample, show_details=True):
        """
        Interpret individual predictions with market insights
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        prediction = self.predict(X_sample)[0]
        probabilities = self.predict_proba(X_sample)[0]

        market_types = ['Buyers Market', 'Even Market', 'Sellers Market']
        predicted_market = market_types[prediction]
        confidence = probabilities[prediction]

        if show_details:
            print(f"🏠 Market Prediction: {predicted_market}")
            print(f"📈 Confidence: {confidence:.2%}")
            print("\nProbability Distribution:")
            for i, (market, prob) in enumerate(zip(market_types, probabilities)):
                print(f"  {market}: {prob:.2%}")

            # Market insights
            sample_data = X_sample.iloc[0] if hasattr(X_sample, 'iloc') else X_sample[0]

            print(f"\n🔍 Key Market Indicators:")
            print(f"  Inventory Months: {sample_data[5]:.1f}")
            print(f"  Absorption Rate: {sample_data[1]:.2f}")
            print(f"  Days on Market: {sample_data[12]:.0f}")
            print(f"  Market Heat Score: {sample_data[8]:.2f}")

        return {
            'prediction': prediction,
            'market_type': predicted_market,
            'confidence': confidence,
            'probabilities': dict(zip(market_types, probabilities))
        }

# Example usage and complete workflow
def main():
    """
    Complete example of using the Housing Market Predictor
    """
    print("🏡 Housing Market Prediction with XGBoost")
    print("=" * 50)

    # Initialize the predictor
    predictor = HousingMarketPredictor()

    # Generate sample data
    print("📊 Generating sample housing market data...")
    raw_data = predictor.create_sample_data(n_samples=2000)
    print(f"Generated {len(raw_data)} samples")

    # Engineer features
    print("🔧 Engineering features...")
    featured_data = predictor.engineer_features(raw_data)

    # Prepare features and target
    X = predictor.prepare_features(featured_data)
    y = featured_data['market_condition']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    predictor.fit(X_train, y_train, optimize_hyperparameters=False)  # Set to True for full optimization

    # Evaluate model
    results = predictor.evaluate_model(X_test, y_test)

    # Feature importance
    importance_df = predictor.get_feature_importance(plot=True)
    print("\nTop 5 Most Important Features:")
    print(importance_df.head())

    # Example prediction
    print("\n" + "="*50)
    print("🔮 Example Market Prediction:")
    sample_data = X_test.iloc[:1]
    actual_market = ['Buyers Market', 'Even Market', 'Sellers Market'][y_test.iloc[0]]

    print(f"Actual Market: {actual_market}")
    prediction_result = predictor.interpret_prediction(sample_data)

    return predictor, results

if __name__ == "__main__":
    # Run the complete example
    model, evaluation_results = main()

    print("\n🎉 Model training and evaluation completed successfully!")
    print(f"Final Model Accuracy: {evaluation_results['accuracy']:.4f}")
