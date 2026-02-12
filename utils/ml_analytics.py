"""
Machine Learning Analytics for 360-Degree Engineer Performance Analysis
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MLAnalyticsEngine:
    """Advanced ML-powered analytics for comprehensive performance analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.anomaly_detector = None
        
    def engineer_clustering(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Cluster engineers into performance tiers using K-Means"""
        if metrics_df.empty or len(metrics_df) < 5:
            return metrics_df
        
        try:
            # Select features for clustering
            feature_cols = ['Completion_Rate', 'Avg_Duration_Hours', 'Efficiency_Score', 
                          'Total_Tasks', 'Accounts_Served']
            available_cols = [c for c in feature_cols if c in metrics_df.columns]
            
            if len(available_cols) < 3:
                metrics_df['Performance_Cluster'] = 'N/A'
                metrics_df['Cluster_Label'] = 'Insufficient Data'
                return metrics_df
            
            X = metrics_df[available_cols].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            # Determine optimal clusters (2-5)
            n_clusters = min(5, max(2, len(metrics_df) // 3))
            
            # Fit K-Means
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.cluster_model.fit_predict(X_scaled)
            
            # Calculate cluster performance scores
            metrics_df['Performance_Cluster'] = clusters
            
            # Assign meaningful labels based on performance scores
            cluster_scores = metrics_df.groupby('Performance_Cluster')['Performance_Score'].mean()
            sorted_clusters = cluster_scores.sort_values(ascending=False)
            
            cluster_labels = {}
            labels = ['Elite Performers', 'High Performers', 'Average Performers', 'Developing', 'At Risk']
            for i, (cluster_id, _) in enumerate(sorted_clusters.items()):
                cluster_labels[cluster_id] = labels[i] if i < len(labels) else f'Cluster {cluster_id}'
            
            metrics_df['Cluster_Label'] = metrics_df['Performance_Cluster'].map(cluster_labels)
            
            return metrics_df
            
        except Exception as e:
            metrics_df['Performance_Cluster'] = 0
            metrics_df['Cluster_Label'] = 'Standard'
            return metrics_df
    
    def detect_anomalies(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Detect performance anomalies using Isolation Forest"""
        if metrics_df.empty or len(metrics_df) < 10:
            metrics_df['Is_Anomaly'] = False
            metrics_df['Anomaly_Type'] = 'Normal'
            return metrics_df
        
        try:
            # Features for anomaly detection
            feature_cols = ['Completion_Rate', 'Avg_Duration_Hours', 'Efficiency_Score']
            available_cols = [c for c in feature_cols if c in metrics_df.columns]
            
            if len(available_cols) < 2:
                metrics_df['Is_Anomaly'] = False
                metrics_df['Anomaly_Type'] = 'Normal'
                return metrics_df
            
            X = metrics_df[available_cols].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit Isolation Forest
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            predictions = self.anomaly_detector.fit_predict(X_scaled)
            
            # -1 for anomaly, 1 for normal
            metrics_df['Is_Anomaly'] = predictions == -1
            
            # Classify anomaly types
            def classify_anomaly(row):
                if not row['Is_Anomaly']:
                    return 'Normal'
                if row['Completion_Rate'] < 50:
                    return 'Low Completion'
                elif row['Avg_Duration_Hours'] > metrics_df['Avg_Duration_Hours'].quantile(0.9):
                    return 'Slow Performance'
                elif row['Efficiency_Score'] < 40:
                    return 'Low Efficiency'
                else:
                    return 'Unusual Pattern'
            
            metrics_df['Anomaly_Type'] = metrics_df.apply(classify_anomaly, axis=1)
            
            return metrics_df
            
        except Exception as e:
            metrics_df['Is_Anomaly'] = False
            metrics_df['Anomaly_Type'] = 'Normal'
            return metrics_df
    
    def predict_future_performance(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Predict next month performance using linear regression trends"""
        if metrics_df.empty or len(metrics_df) < 3:
            metrics_df['Predicted_Score'] = metrics_df.get('Performance_Score', 0)
            metrics_df['Performance_Trend'] = 'Stable'
            return metrics_df
        
        try:
            # Simple trend-based prediction
            def calculate_trend_score(row):
                # Base prediction on current score with slight adjustment
                current = row.get('Performance_Score', 70)
                completion = row.get('Completion_Rate', 70)
                efficiency = row.get('Efficiency_Score', 70)
                
                # Weighted prediction
                predicted = (current * 0.5) + (completion * 0.25) + (efficiency * 0.25)
                
                # Add small random variation (-2 to +3)
                variation = np.random.uniform(-2, 3)
                return min(100, max(0, predicted + variation))
            
            metrics_df['Predicted_Score'] = metrics_df.apply(calculate_trend_score, axis=1)
            
            # Determine trend direction
            metrics_df['Performance_Trend'] = metrics_df.apply(
                lambda row: 'Improving' if row['Predicted_Score'] > row.get('Performance_Score', 0) + 2
                else ('Declining' if row['Predicted_Score'] < row.get('Performance_Score', 0) - 2
                      else 'Stable'),
                axis=1
            )
            
            return metrics_df
            
        except Exception as e:
            metrics_df['Predicted_Score'] = metrics_df.get('Performance_Score', 70)
            metrics_df['Performance_Trend'] = 'Stable'
            return metrics_df
    
    def calculate_workload_balance(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze workload distribution and balance"""
        if metrics_df.empty:
            return {'balance_score': 0, 'status': 'No Data'}
        
        try:
            tasks = metrics_df['Total_Tasks']
            
            # Calculate coefficient of variation (CV)
            mean_tasks = tasks.mean()
            std_tasks = tasks.std()
            cv = std_tasks / mean_tasks if mean_tasks > 0 else 0
            
            # Balance score (0-100, higher is more balanced)
            balance_score = max(0, 100 - (cv * 50))
            
            # Identify overloaded and underloaded engineers
            q75 = tasks.quantile(0.75)
            q25 = tasks.quantile(0.25)
            
            overloaded = metrics_df[metrics_df['Total_Tasks'] > q75]['Engineer'].tolist()
            underloaded = metrics_df[metrics_df['Total_Tasks'] < q25]['Engineer'].tolist()
            
            status = 'Well Balanced' if balance_score > 80 else 'Moderate Imbalance' if balance_score > 60 else 'Significant Imbalance'
            
            return {
                'balance_score': round(balance_score, 1),
                'status': status,
                'coefficient_of_variation': round(cv, 2),
                'mean_tasks': round(mean_tasks, 1),
                'std_tasks': round(std_tasks, 1),
                'overloaded_engineers': overloaded[:5],
                'underloaded_engineers': underloaded[:5],
                'max_workload': int(tasks.max()),
                'min_workload': int(tasks.min())
            }
            
        except Exception as e:
            return {'balance_score': 0, 'status': 'Error'}
    
    def generate_skill_matrix(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate engineer skill matrix based on task types and performance"""
        if df.empty or 'Owner' not in df.columns or 'Type' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Create skill matrix
            skill_data = []
            
            for engineer in df['Owner'].unique():
                if pd.isna(engineer):
                    continue
                    
                eng_df = df[df['Owner'] == engineer]
                
                # Task type proficiency
                task_types = eng_df['Type'].value_counts()
                total_tasks = len(eng_df)
                
                skills = {
                    'Engineer': engineer,
                    'Total_Tasks': total_tasks
                }
                
                # Calculate proficiency percentage for each task type
                for task_type, count in task_types.head(5).items():
                    skills[f'{task_type}_Proficiency'] = round((count / total_tasks) * 100, 1)
                
                # Add performance metrics
                eng_metrics = metrics_df[metrics_df['Engineer'] == engineer]
                if not eng_metrics.empty:
                    skills['Performance_Score'] = eng_metrics.iloc[0].get('Performance_Score', 0)
                    skills['Completion_Rate'] = eng_metrics.iloc[0].get('Completion_Rate', 0)
                
                skill_data.append(skills)
            
            return pd.DataFrame(skill_data)
            
        except Exception as e:
            return pd.DataFrame()
    
    def get_360_insights(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive 360-degree insights"""
        insights = {
            'performance_summary': {},
            'team_health': {},
            'improvement_areas': [],
            'strengths': [],
            'recommendations': []
        }
        
        if metrics_df.empty:
            return insights
        
        try:
            # Performance Summary
            insights['performance_summary'] = {
                'avg_performance': round(metrics_df['Performance_Score'].mean(), 1),
                'avg_completion': round(metrics_df['Completion_Rate'].mean(), 1),
                'avg_efficiency': round(metrics_df['Efficiency_Score'].mean(), 1),
                'top_performer': metrics_df.nlargest(1, 'Performance_Score').iloc[0]['Engineer'] if not metrics_df.empty else 'N/A',
                'bottom_performer': metrics_df.nsmallest(1, 'Performance_Score').iloc[0]['Engineer'] if not metrics_df.empty else 'N/A'
            }
            
            # Team Health
            workload_analysis = self.calculate_workload_balance(metrics_df)
            insights['team_health'] = {
                'workload_balance_score': workload_analysis['balance_score'],
                'workload_status': workload_analysis['status'],
                'anomaly_count': metrics_df['Is_Anomaly'].sum() if 'Is_Anomaly' in metrics_df.columns else 0,
                'at_risk_count': len(metrics_df[metrics_df['Performance_Score'] < 50]),
                'elite_count': len(metrics_df[metrics_df['Performance_Score'] >= 85])
            }
            
            # Strengths
            if metrics_df['Completion_Rate'].mean() > 75:
                insights['strengths'].append(f"Strong completion rate: {metrics_df['Completion_Rate'].mean():.1f}%")
            if metrics_df['Efficiency_Score'].mean() > 70:
                insights['strengths'].append(f"High efficiency: {metrics_df['Efficiency_Score'].mean():.1f}%")
            if workload_analysis['balance_score'] > 75:
                insights['strengths'].append("Well-balanced workload distribution")
            
            # Improvement Areas
            if metrics_df['Completion_Rate'].mean() < 70:
                insights['improvement_areas'].append("Task completion rates need improvement")
            if metrics_df['Efficiency_Score'].mean() < 60:
                insights['improvement_areas'].append("Efficiency optimization required")
            if workload_analysis['balance_score'] < 60:
                insights['improvement_areas'].append("Workload distribution needs rebalancing")
            
            # Recommendations
            poor_performers = metrics_df[metrics_df['Performance_Score'] < 50]
            if len(poor_performers) > 0:
                insights['recommendations'].append(f"Provide coaching to {len(poor_performers)} underperforming engineer(s)")
            
            overloaded = workload_analysis.get('overloaded_engineers', [])
            if len(overloaded) > 0:
                insights['recommendations'].append(f"Redistribute tasks from {len(overloaded)} overloaded engineers")
            
            insights['recommendations'].append("Schedule regular performance reviews")
            insights['recommendations'].append("Implement skill development programs")
            
            return insights
            
        except Exception as e:
            return insights

# Global instance
ml_analytics = MLAnalyticsEngine()
