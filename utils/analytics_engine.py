"""
Analytics engine for Engineer Performance Dashboard
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from config.settings import PERFORMANCE_WEIGHTS

class AnalyticsEngine:
    """Handles all analytics and metric calculations"""
    
    def __init__(self):
        self.metrics_cache = {}
        
    def calculate_engineer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive engineer performance metrics"""
        
        if 'Owner' not in df.columns:
            return pd.DataFrame()
        
        metrics_list = []
        
        for engineer in df['Owner'].unique():
            if pd.isna(engineer) or engineer == 'Not Specified':
                continue
                
            eng_df = df[df['Owner'] == engineer]
            
            # Skip engineers with no data
            if len(eng_df) == 0:
                continue
            
            # Basic metrics
            total_tasks = len(eng_df)
            
            # Completion metrics
            completed_tasks = len(eng_df[
                eng_df['Status_Category'].str.contains('DONE', case=False, na=False)
            ]) if 'Status_Category' in eng_df.columns else 0
            
            completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            
            # Time metrics
            avg_duration = eng_df['Duration_Hours'].mean() if 'Duration_Hours' in eng_df.columns else 0
            median_duration = eng_df['Duration_Hours'].median() if 'Duration_Hours' in eng_df.columns else 0
            total_hours = eng_df['Duration_Hours'].sum() if 'Duration_Hours' in eng_df.columns else 0
            
            # Task type distribution
            task_types = eng_df['Type'].value_counts().to_dict() if 'Type' in eng_df.columns else {}
            primary_task = max(task_types.items(), key=lambda x: x[1])[0] if task_types else 'None'
            
            # Priority metrics
            high_priority = len(eng_df[eng_df['Priority_Order'] <= 2]) if 'Priority_Order' in eng_df.columns else 0
            asap_tasks = len(eng_df[eng_df['Priority'] == '1-ASAP']) if 'Priority' in eng_df.columns else 0
            
            # Account metrics
            accounts_served = eng_df['Account'].nunique() if 'Account' in eng_df.columns else 0
            primary_account = eng_df['Account'].mode().iloc[0] if accounts_served > 0 else 'None'
            
            # Time-based metrics
            if 'Date' in eng_df.columns:
                active_days = eng_df['Date'].nunique()
                tasks_per_day = total_tasks / active_days if active_days > 0 else 0
            else:
                active_days = 0
                tasks_per_day = 0
            
            # Status distribution
            status_counts = eng_df['Status_Category'].value_counts().to_dict() if 'Status_Category' in eng_df.columns else {}
            
            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(eng_df)
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(
                completion_rate=completion_rate,
                efficiency_score=efficiency_score,
                high_priority_tasks=high_priority,
                total_tasks=total_tasks,
                accounts_served=accounts_served
            )
            
            metrics_list.append({
                'Engineer': engineer,
                'Total_Tasks': total_tasks,
                'Completed_Tasks': completed_tasks,
                'Completion_Rate': round(completion_rate, 2),
                'Avg_Duration_Hours': round(avg_duration, 2),
                'Median_Duration_Hours': round(median_duration, 2),
                'Total_Work_Hours': round(total_hours, 2),
                'High_Priority_Tasks': high_priority,
                'ASAP_Tasks': asap_tasks,
                'Task_Types': len(task_types),
                'Primary_Task_Type': primary_task,
                'Accounts_Served': accounts_served,
                'Primary_Account': primary_account,
                'Active_Days': active_days,
                'Tasks_Per_Day': round(tasks_per_day, 2),
                'Efficiency_Score': round(efficiency_score, 2),
                'Performance_Score': round(performance_score, 2),
                'Status_Distribution': status_counts,
                'Task_Type_Distribution': task_types
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        if not metrics_df.empty:
            # Add rankings
            metrics_df['Rank_Tasks'] = metrics_df['Total_Tasks'].rank(ascending=False, method='min')
            metrics_df['Rank_Efficiency'] = metrics_df['Efficiency_Score'].rank(ascending=False, method='min')
            metrics_df['Rank_Performance'] = metrics_df['Performance_Score'].rank(ascending=False, method='min')
            
            # Calculate percentiles
            metrics_df['Percentile_Tasks'] = metrics_df['Total_Tasks'].rank(pct=True) * 100
            metrics_df['Percentile_Performance'] = metrics_df['Performance_Score'].rank(pct=True) * 100
        
        return metrics_df
    
    def _calculate_efficiency_score(self, df: pd.DataFrame) -> float:
        """Calculate efficiency score based on duration and completion"""
        if len(df) == 0 or 'Duration_Hours' not in df.columns:
            return 50
        
        # Get completed tasks only
        completed_df = df[
            df['Status_Category'].str.contains('DONE', case=False, na=False)
        ] if 'Status_Category' in df.columns else df
        
        if len(completed_df) == 0:
            return 50
        
        # Calculate average duration and compare to overall average
        avg_duration = completed_df['Duration_Hours'].mean()
        
        # Score: lower duration = higher efficiency
        if avg_duration <= 1:
            efficiency = 95
        elif avg_duration <= 2:
            efficiency = 85
        elif avg_duration <= 4:
            efficiency = 75
        elif avg_duration <= 8:
            efficiency = 65
        else:
            efficiency = 50
        
        # Adjust based on task volume
        volume_bonus = min(20, len(completed_df) / 10)
        
        return min(100, efficiency + volume_bonus)
    
    def _calculate_performance_score(self, completion_rate: float, efficiency_score: float,
                                   high_priority_tasks: int, total_tasks: int,
                                   accounts_served: int) -> float:
        """Calculate weighted performance score"""
        
        # Normalize high priority handling
        high_priority_ratio = (high_priority_tasks / total_tasks * 100) if total_tasks > 0 else 0
        high_priority_score = min(100, high_priority_ratio * 2)
        
        # Normalize accounts served
        accounts_score = min(100, accounts_served * 20)
        
        # Weighted calculation
        score = (
            completion_rate * PERFORMANCE_WEIGHTS['completion_rate'] +
            efficiency_score * PERFORMANCE_WEIGHTS['efficiency'] +
            high_priority_score * PERFORMANCE_WEIGHTS['high_priority'] +
            accounts_score * PERFORMANCE_WEIGHTS['accounts_served']
        )
        
        return min(100, max(0, score))
    
    def calculate_temporal_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate time-based metrics and trends"""
        
        metrics = {}
        
        if 'Date' in df.columns and 'Duration_Hours' in df.columns:
            # Daily trends
            daily_stats = df.groupby('Date').agg({
                'Duration_Hours': ['count', 'sum', 'mean'],
                'Priority_Order': 'mean'
            }).round(2)
            
            metrics['daily_tasks'] = daily_stats[('Duration_Hours', 'count')].tolist()
            metrics['daily_hours'] = daily_stats[('Duration_Hours', 'sum')].tolist()
            
            # Weekly trends
            if 'Week' not in df.columns:
                df['Week'] = pd.to_datetime(df['Date']).dt.isocalendar().week
            
            weekly_stats = df.groupby('Week').agg({
                'Duration_Hours': 'count',
                'Owner': 'nunique'
            }).round(2)
            
            metrics['weekly_tasks'] = weekly_stats['Duration_Hours'].tolist()
            metrics['weekly_engineers'] = weekly_stats['Owner'].tolist()
            
            # Peak hours
            if 'Hour' in df.columns:
                peak_hours = df['Hour'].value_counts().head(3).to_dict()
                metrics['peak_hours'] = peak_hours
            
            # Busiest days
            if 'DayOfWeek' in df.columns:
                busy_days = df['DayOfWeek'].value_counts().head(3).to_dict()
                metrics['busy_days'] = busy_days
        
        return metrics
    
    def calculate_task_type_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics by task type"""
        
        if 'Type' not in df.columns:
            return pd.DataFrame()
        
        type_metrics = []
        
        for task_type in df['Type'].unique():
            if pd.isna(task_type):
                continue
                
            type_df = df[df['Type'] == task_type]
            
            metrics = {
                'Task_Type': task_type,
                'Total_Tasks': len(type_df),
                'Unique_Engineers': type_df['Owner'].nunique() if 'Owner' in type_df.columns else 0,
                'Unique_Accounts': type_df['Account'].nunique() if 'Account' in type_df.columns else 0,
                'Avg_Duration': type_df['Duration_Hours'].mean() if 'Duration_Hours' in type_df.columns else 0,
                'Completion_Rate': (
                    len(type_df[type_df['Status_Category'].str.contains('DONE', case=False, na=False)]) /
                    len(type_df) * 100
                ) if 'Status_Category' in type_df.columns else 0
            }
            
            type_metrics.append(metrics)
        
        return pd.DataFrame(type_metrics).sort_values('Total_Tasks', ascending=False)
    
    def generate_performance_insights(self, metrics_df: pd.DataFrame) -> List[str]:
        """Generate natural language insights from metrics"""
        
        insights = []
        
        if metrics_df.empty:
            return insights
        
        # Top performer insights
        top_performer = metrics_df.nlargest(1, 'Performance_Score').iloc[0]
        insights.append(
            f"🏆 **Top Performer**: {top_performer['Engineer']} "
            f"with {top_performer['Performance_Score']:.1f}% performance score, "
            f"completed {top_performer['Completed_Tasks']} tasks at {top_performer['Completion_Rate']:.1f}% rate"
        )
        
        # Completion rate insights
        avg_completion = metrics_df['Completion_Rate'].mean()
        if avg_completion > 85:
            insights.append(f"✅ **Excellent Team Performance**: Average completion rate is {avg_completion:.1f}%")
        elif avg_completion < 60:
            insights.append(f"⚠️ **Improvement Needed**: Average completion rate is only {avg_completion:.1f}%")
        
        # Efficiency insights
        avg_efficiency = metrics_df['Efficiency_Score'].mean()
        if avg_efficiency > 75:
            insights.append(f"⚡ **High Efficiency**: Team average efficiency score is {avg_efficiency:.1f}%")
        
        # Workload insights
        avg_tasks = metrics_df['Total_Tasks'].mean()
        max_tasks = metrics_df['Total_Tasks'].max()
        min_tasks = metrics_df['Total_Tasks'].min()
        
        if max_tasks > avg_tasks * 1.5:
            overloaded = metrics_df[metrics_df['Total_Tasks'] > avg_tasks * 1.5]
            insights.append(
                f"⚠️ **Workload Imbalance**: {len(overloaded)} engineer(s) have significantly higher workload "
                f"({overloaded.iloc[0]['Engineer']}: {overloaded.iloc[0]['Total_Tasks']} tasks)"
            )
        
        # Account coverage insights
        avg_accounts = metrics_df['Accounts_Served'].mean()
        top_account_engineer = metrics_df.nlargest(1, 'Accounts_Served').iloc[0]
        insights.append(
            f"🌐 **Account Coverage**: Engineers serve average {avg_accounts:.1f} accounts each. "
            f"{top_account_engineer['Engineer']} serves {int(top_account_engineer['Accounts_Served'])} accounts"
        )
        
        return insights

# Global instance
analytics_engine = AnalyticsEngine()