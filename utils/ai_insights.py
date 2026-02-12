"""
Gemini AI integration for intelligent insights
"""
import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any
from config.settings import APP_TITLE

# Load environment variables
load_dotenv()

class AIInsightGenerator:
    """Handles Gemini AI-powered insights generation"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model with API key from .env file"""
        try:
            import google.generativeai as genai
            
            # Load API key from environment variable (.env file)
            api_key = os.getenv('GEMINI_API_KEY')
            
            if not api_key:
                st.warning("⚠️ GEMINI_API_KEY not found in .env file. AI insights will be disabled.")
                return
            
            genai.configure(api_key=api_key)
            # Use Gemini 2.5 Flash model
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ Gemini 2.5 Flash model initialized successfully")
                
        except ImportError:
            st.warning("⚠️ Google Generative AI package not installed. AI insights will be disabled.")
        except Exception as e:
            st.warning(f"⚠️ Failed to initialize Gemini: {str(e)}")
    
    def generate_performance_summary(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
        """Generate comprehensive performance summary using AI"""
        
        if self.model is None:
            return self._get_fallback_insights(metrics_df)
        
        try:
            # Prepare data summary
            summary = self._prepare_data_summary(df, metrics_df)
            
            prompt = f"""
            You are a senior performance analyst. Analyze this field engineer performance data and provide insights:
            
            📊 OVERALL METRICS:
            - Time Period: {summary['date_range']}
            - Total Engineers: {summary['total_engineers']}
            - Total Tasks: {summary['total_tasks']}
            - Completion Rate: {summary['completion_rate']:.1f}%
            - Average Task Duration: {summary['avg_duration']:.1f} hours
            
            🏆 TOP PERFORMERS:
            {self._format_dict(summary['top_performers'])}
            
            📋 TASK BREAKDOWN:
            {self._format_dict(summary['task_breakdown'])}
            
            ⚡ PRIORITY DISTRIBUTION:
            {self._format_dict(summary['priority_distribution'])}
            
            Please provide:
            1. THREE key strengths of the team (with specific metrics)
            2. TWO critical areas needing improvement (with actionable recommendations)
            3. ONE risk factor to monitor
            4. Overall team performance rating (Excellent/Good/Fair/Needs Improvement)
            
            Format the response with clear sections and bullet points. Be specific and data-driven.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"⚠️ Error generating AI insights: {str(e)}\n\n{self._get_fallback_insights(metrics_df)}"
    
    def generate_engineer_feedback(self, engineer_data: pd.Series) -> str:
        """Generate personalized feedback for an engineer"""
        
        if self.model is None:
            return self._get_fallback_feedback(engineer_data)
        
        try:
            prompt = f"""
            Generate personalized professional feedback for this field engineer based on their performance metrics:
            
            Engineer: {engineer_data.get('Engineer', 'Unknown')}
            - Total Tasks: {engineer_data.get('Total_Tasks', 0)}
            - Completion Rate: {engineer_data.get('Completion_Rate', 0):.1f}%
            - Performance Score: {engineer_data.get('Performance_Score', 0):.1f}%
            - Efficiency Score: {engineer_data.get('Efficiency_Score', 0):.1f}%
            - Average Task Duration: {engineer_data.get('Avg_Duration_Hours', 0):.1f} hours
            - High Priority Tasks: {engineer_data.get('High_Priority_Tasks', 0)}
            - Accounts Served: {engineer_data.get('Accounts_Served', 0)}
            - Primary Task Type: {engineer_data.get('Primary_Task_Type', 'N/A')}
            
            Provide:
            1. ONE sentence praising their strength
            2. ONE constructive suggestion for improvement
            3. ONE specific goal for next month
            
            Keep it professional, positive, and actionable.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except:
            return self._get_fallback_feedback(engineer_data)
    
    def predict_performance_trend(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict future performance trends"""
        
        predictions = {
            'trend': 'Stable',
            'confidence': 70,
            'next_month_tasks': 0,
            'top_potential': [],
            'at_risk': []
        }
        
        if len(metrics_df) < 3:
            return predictions
        
        try:
            # Calculate trends
            avg_performance = metrics_df['Performance_Score'].mean()
            performance_trend = metrics_df['Performance_Score'].iloc[-3:].mean() - metrics_df['Performance_Score'].iloc[:3].mean()
            
            if performance_trend > 5:
                predictions['trend'] = 'Improving'
                predictions['confidence'] = 80
            elif performance_trend < -5:
                predictions['trend'] = 'Declining'
                predictions['confidence'] = 75
            else:
                predictions['trend'] = 'Stable'
                predictions['confidence'] = 85
            
            # Predict next month tasks
            avg_daily_tasks = metrics_df['Total_Tasks'].sum() / 30  # Assuming monthly
            predictions['next_month_tasks'] = int(avg_daily_tasks * 22)  # ~22 working days
            
            # Identify top potential engineers
            predictions['top_potential'] = metrics_df.nlargest(3, 'Efficiency_Score')[
                ['Engineer', 'Efficiency_Score', 'Performance_Score']
            ].to_dict('records')
            
            # Identify at-risk engineers
            at_risk = metrics_df[
                (metrics_df['Completion_Rate'] < 50) |
                (metrics_df['Performance_Score'] < 40)
            ]
            predictions['at_risk'] = at_risk[['Engineer', 'Completion_Rate', 'Performance_Score']].head(3).to_dict('records')
            
        except:
            pass
        
        return predictions
    
    def _prepare_data_summary(self, df: pd.DataFrame, metrics_df: pd.DataFrame) -> Dict:
        """Prepare data summary for AI prompts"""
        
        summary = {
            'total_tasks': len(df),
            'total_engineers': len(metrics_df) if not metrics_df.empty else 0,
            'completion_rate': metrics_df['Completion_Rate'].mean() if not metrics_df.empty else 0,
            'avg_duration': df['Duration_Hours'].mean() if 'Duration_Hours' in df.columns else 0,
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else 'N/A',
            'top_performers': {},
            'task_breakdown': {},
            'priority_distribution': {}
        }
        
        if not metrics_df.empty:
            top_3 = metrics_df.nlargest(3, 'Performance_Score')
            for _, row in top_3.iterrows():
                summary['top_performers'][row['Engineer']] = f"{row['Performance_Score']:.1f}%"
        
        if 'Type' in df.columns:
            task_types = df['Type'].value_counts().head(5).to_dict()
            summary['task_breakdown'] = {k: int(v) for k, v in task_types.items()}
        
        if 'Priority' in df.columns:
            priorities = df['Priority'].value_counts().head(5).to_dict()
            summary['priority_distribution'] = {k: int(v) for k, v in priorities.items()}
        
        return summary
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for prompt"""
        if not d:
            return "None"
        return '\n'.join([f"  - {k}: {v}" for k, v in d.items()])
    
    def _get_fallback_insights(self, metrics_df: pd.DataFrame) -> str:
        """Fallback insights when AI is unavailable"""
        
        insights = []
        
        if metrics_df.empty:
            return "No performance data available for analysis."
        
        # Basic statistical insights
        avg_completion = metrics_df['Completion_Rate'].mean()
        avg_efficiency = metrics_df['Efficiency_Score'].mean()
        avg_performance = metrics_df['Performance_Score'].mean()
        
        insights.append("📊 **Team Performance Summary**")
        insights.append("")
        insights.append(f"• **Completion Rate**: {avg_completion:.1f}% - {'Excellent' if avg_completion > 85 else 'Good' if avg_completion > 70 else 'Fair' if avg_completion > 50 else 'Needs Improvement'}")
        insights.append(f"• **Efficiency Score**: {avg_efficiency:.1f}%")
        insights.append(f"• **Overall Performance**: {avg_performance:.1f}%")
        insights.append("")
        
        # Top performer
        if len(metrics_df) > 0:
            top = metrics_df.nlargest(1, 'Performance_Score').iloc[0]
            insights.append(f"🏆 **Top Performer**: {top['Engineer']}")
            insights.append(f"   - Tasks: {top['Total_Tasks']}, Completion: {top['Completion_Rate']:.1f}%")
            insights.append("")
        
        # Recommendations
        insights.append("💡 **Recommendations**")
        insights.append("")
        
        if avg_completion < 70:
            insights.append("• Focus on improving task completion rates")
        if avg_efficiency < 60:
            insights.append("• Review and optimize task workflows")
        
        poor_performers = metrics_df[metrics_df['Performance_Score'] < 50]
        if len(poor_performers) > 0:
            insights.append(f"• Provide coaching for {len(poor_performers)} engineer(s) with low performance")
        
        return '\n'.join(insights)
    
    def _get_fallback_feedback(self, engineer_data: pd.Series) -> str:
        """Fallback feedback when AI is unavailable"""
        
        feedback = []
        feedback.append(f"**Performance Feedback for {engineer_data.get('Engineer', 'Unknown')}**")
        feedback.append("")
        
        # Praise
        completion = engineer_data.get('Completion_Rate', 0)
        if completion > 85:
            feedback.append("✅ **Strength**: Excellent task completion rate")
        elif completion > 70:
            feedback.append("✅ **Strength**: Good task completion record")
        else:
            feedback.append("✅ **Strength**: Active and engaged team member")
        
        # Improvement
        feedback.append("")
        feedback.append("📈 **Suggestion for Improvement**:")
        
        if completion < 70:
            feedback.append("• Focus on improving task completion rate")
        if engineer_data.get('Avg_Duration_Hours', 0) > 4:
            feedback.append("• Work on reducing average task completion time")
        if engineer_data.get('Accounts_Served', 0) < 2:
            feedback.append("• Seek opportunities to work with different accounts")
        
        # Goal
        feedback.append("")
        feedback.append("🎯 **Next Month Goal**:")
        feedback.append(f"• Achieve {min(95, completion + 10):.0f}% completion rate")
        
        return '\n'.join(feedback)

# Global instance
ai_insight_generator = AIInsightGenerator()