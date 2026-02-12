"""
Visualization utilities for Engineer Performance Dashboard
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

class ChartGenerator:
    """Handles all chart generation and styling"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1E3A8A',
            'secondary': '#2563EB',
            'success': '#10B981',
            'warning': '#FBBF24',
            'danger': '#EF4444',
            'info': '#3B82F6',
            'purple': '#8B5CF6',
            'pink': '#EC4899',
            'gray': '#6B7280'
        }
        
        self.color_sequences = {
            'qualitative': px.colors.qualitative.Set3,
            'sequential': px.colors.sequential.Blues,
            'diverging': px.colors.diverging.RdBu
        }
        # default theme for chart styling: 'dark' or 'light'
        self.theme = 'dark'
    
    def apply_common_layout(self, fig: go.Figure, title: str = None, 
                           height: int = 400, show_legend: bool = True) -> go.Figure:
        """Apply common layout settings to figures"""
        # Theme-aware colors
        if getattr(self, 'theme', 'dark') == 'dark':
            title_color = '#E2E8F0'  # light
            font_color = '#E2E8F0'
            paper_bg = 'rgba(0,0,0,0)'
            plot_bg = 'rgba(0,0,0,0)'
            axis_color = '#CBD5E1'
        else:
            title_color = '#0f172a'
            font_color = '#0f172a'
            paper_bg = 'white'
            plot_bg = '#F8FAFC'
            axis_color = '#1E293B'

        title_text = title or ''
        fig.update_layout(
            title={
                'text': title_text,
                'font': {'size': 16, 'color': title_color, 'family': 'Arial'},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=height,
            showlegend=show_legend,
            legend={
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': -0.2,
                'xanchor': 'center',
                'x': 0.5,
                'font': {'color': font_color}
            },
            margin={'l': 50, 'r': 50, 't': 80, 'b': 80},
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            font={'family': 'Arial', 'size': 12, 'color': font_color}
        )

        # Ensure axes use readable colors
        fig.update_xaxes(tickfont=dict(color=axis_color), title_font=dict(color=axis_color))
        fig.update_yaxes(tickfont=dict(color=axis_color), title_font=dict(color=axis_color))

        return fig
    
    def create_performance_ranking_chart(self, metrics_df: pd.DataFrame, 
                                        top_n: int = 10) -> go.Figure:
        """Create engineer performance ranking visualization"""
        
        if metrics_df.empty:
            return go.Figure()
        
        df = metrics_df.head(top_n).copy()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Task Volume', 'Performance Score'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Task volume
        fig.add_trace(
            go.Bar(
                x=df['Engineer'],
                y=df['Total_Tasks'],
                name='Total Tasks',
                marker_color=self.color_palette['info'],
                text=df['Total_Tasks'],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Performance score
        fig.add_trace(
            go.Bar(
                x=df['Engineer'],
                y=df['Performance_Score'],
                name='Performance Score',
                marker_color=df['Performance_Score'],
                marker_colorscale='Viridis',
                text=df['Performance_Score'].round(1),
                textposition='outside',
                texttemplate='%{text}%'
            ),
            row=1, col=2
        )
        
        fig.update_yaxes(title_text="Number of Tasks", row=1, col=1)
        fig.update_yaxes(title_text="Score (%)", row=1, col=2)
        
        return self.apply_common_layout(fig, height=500, show_legend=False)
    
    def create_task_type_distribution(self, df: pd.DataFrame, top_n: int = 10) -> go.Figure:
        """Create task type distribution pie chart"""
        
        if 'Type' not in df.columns:
            return go.Figure()
        
        type_counts = df['Type'].value_counts().head(top_n).reset_index()
        type_counts.columns = ['Task Type', 'Count']
        
        fig = px.pie(
            type_counts,
            values='Count',
            names='Task Type',
            color_discrete_sequence=self.color_sequences['qualitative']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value',
            marker=dict(line=dict(color='white', width=2))
        )
        
        return self.apply_common_layout(fig, height=450)
    
    def create_status_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create status distribution bar chart"""
        
        if 'Status_Category' not in df.columns:
            return go.Figure()
        
        status_counts = df['Status_Category'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        colors = []
        for status in status_counts['Status']:
            if 'DONE' in status:
                colors.append(self.color_palette['success'])
            elif 'PROGRESS' in status:
                colors.append(self.color_palette['warning'])
            elif 'PENDING' in status:
                colors.append(self.color_palette['danger'])
            else:
                colors.append(self.color_palette['gray'])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=status_counts['Status'],
            y=status_counts['Count'],
            marker_color=colors,
            text=status_counts['Count'],
            textposition='outside'
        ))
        
        return self.apply_common_layout(fig, height=400)
    
    def create_duration_histogram(self, df: pd.DataFrame) -> go.Figure:
        """Create task duration distribution histogram"""
        
        if 'Duration_Hours' not in df.columns:
            return go.Figure()
        
        # Filter outliers (tasks > 24 hours)
        duration_data = df[df['Duration_Hours'] <= 24]['Duration_Hours']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=duration_data,
            nbinsx=30,
            marker_color=self.color_palette['info'],
            marker_line_color='white',
            marker_line_width=1
        ))
        
        # Add mean line
        mean_val = duration_data.mean()
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color=self.color_palette['danger'],
            annotation_text=f"Mean: {mean_val:.1f}h",
            annotation_position="top right"
        )
        
        fig.update_xaxes(title_text="Duration (Hours)")
        fig.update_yaxes(title_text="Number of Tasks")
        
        return self.apply_common_layout(fig, height=400)
    
    def create_daily_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create daily task volume trend line chart"""
        
        if 'Date' not in df.columns:
            return go.Figure()
        
        daily_tasks = df.groupby('Date').size().reset_index(name='Count')
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_tasks['Date'],
            y=daily_tasks['Count'],
            mode='lines+markers',
            name='Daily Tasks',
            line=dict(color=self.color_palette['secondary'], width=3),
            marker=dict(size=8, color=self.color_palette['primary'])
        ))
        
        # Add trend line
        if len(daily_tasks) > 1:
            x_numeric = np.arange(len(daily_tasks))
            z = np.polyfit(x_numeric, daily_tasks['Count'], 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=daily_tasks['Date'],
                y=p(x_numeric),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color=self.color_palette['gray'])
            ))
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Number of Tasks")
        
        return self.apply_common_layout(fig, height=400)
    
    def create_workload_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create workload distribution box plot"""
        
        if 'Owner' not in df.columns:
            return go.Figure()
        
        workload = df['Owner'].value_counts().reset_index()
        workload.columns = ['Engineer', 'Task Count']
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=workload['Task Count'],
            name='Workload Distribution',
            boxmean='sd',
            marker_color=self.color_palette['info'],
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig.update_yaxes(title_text="Number of Tasks")
        
        # Add statistics annotations
        q1 = workload['Task Count'].quantile(0.25)
        q3 = workload['Task Count'].quantile(0.75)
        median = workload['Task Count'].median()
        
        fig.add_annotation(
            x=0.5, y=median,
            text=f"Median: {median:.0f}",
            showarrow=False,
            yshift=10,
            font=dict(color='white', size=11)
        )
        
        return self.apply_common_layout(fig, height=400)
    
    def create_priority_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create priority distribution chart"""
        
        if 'Priority' not in df.columns:
            return go.Figure()
        
        priority_counts = df['Priority'].value_counts().reset_index()
        priority_counts.columns = ['Priority', 'Count']
        
        # Sort by priority order
        priority_order = ['1-ASAP', '2-High', '3-Medium', '4-Low', 'Not Specified']
        priority_counts['Priority'] = pd.Categorical(
            priority_counts['Priority'],
            categories=priority_order,
            ordered=True
        )
        priority_counts = priority_counts.sort_values('Priority')
        
        colors = [
            self.color_palette['danger'],
            self.color_palette['warning'],
            self.color_palette['info'],
            self.color_palette['success'],
            self.color_palette['gray']
        ]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=priority_counts['Priority'],
            y=priority_counts['Count'],
            marker_color=colors[:len(priority_counts)],
            text=priority_counts['Count'],
            textposition='outside'
        ))
        
        fig.update_xaxes(title_text="Priority Level")
        fig.update_yaxes(title_text="Number of Tasks")
        
        return self.apply_common_layout(fig, height=400)
    
    def create_account_distribution(self, df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """Create account distribution horizontal bar chart"""
        
        if 'Account' not in df.columns:
            return go.Figure()
        
        account_counts = df['Account'].value_counts().head(top_n).reset_index()
        account_counts.columns = ['Account', 'Task Count']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=account_counts['Task Count'],
            y=account_counts['Account'],
            orientation='h',
            marker_color=account_counts['Task Count'],
            marker_colorscale='Blues',
            text=account_counts['Task Count'],
            textposition='outside'
        ))
        
        fig.update_xaxes(title_text="Number of Tasks")
        fig.update_yaxes(title_text="", autorange="reversed")
        
        return self.apply_common_layout(fig, height=500)
    
    def create_performance_matrix(self, metrics_df: pd.DataFrame) -> go.Figure:
        """Create performance matrix scatter plot"""
        
        if metrics_df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics_df['Total_Tasks'],
            y=metrics_df['Completion_Rate'],
            mode='markers+text',
            marker=dict(
                size=metrics_df['Performance_Score'] / 2 + 10,
                color=metrics_df['Performance_Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Performance<br>Score", x=1.02)
            ),
            text=metrics_df['Engineer'],
            textposition='top center',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Tasks: %{x}<br>' +
                'Completion: %{y:.1f}%<br>' +
                'Performance: %{marker.color:.1f}%<br>' +
                'Efficiency: %{customdata[0]:.1f}%<br>' +
                'Accounts: %{customdata[1]}<extra></extra>'
            ),
            customdata=metrics_df[['Efficiency_Score', 'Accounts_Served']]
        ))
        
        # Add quadrant lines
        avg_tasks = metrics_df['Total_Tasks'].mean()
        avg_completion = metrics_df['Completion_Rate'].mean()
        
        fig.add_hline(y=avg_completion, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=avg_tasks, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant labels
        fig.add_annotation(
            x=max(metrics_df['Total_Tasks']), y=100,
            text="STARS",
            showarrow=False,
            font=dict(size=14, color=self.color_palette['success']),
            xanchor='right', yanchor='top'
        )
        
        fig.add_annotation(
            x=0, y=100,
            text="RISING",
            showarrow=False,
            font=dict(size=14, color=self.color_palette['warning']),
            xanchor='left', yanchor='top'
        )
        
        fig.add_annotation(
            x=max(metrics_df['Total_Tasks']), y=0,
            text="WORKHORSES",
            showarrow=False,
            font=dict(size=14, color=self.color_palette['info']),
            xanchor='right', yanchor='bottom'
        )
        
        fig.add_annotation(
            x=0, y=0,
            text="NEEDS FOCUS",
            showarrow=False,
            font=dict(size=14, color=self.color_palette['danger']),
            xanchor='left', yanchor='bottom'
        )
        
        fig.update_xaxes(title_text="Total Tasks")
        fig.update_yaxes(title_text="Completion Rate (%)")
        
        return self.apply_common_layout(fig, height=600)

# Global instance
chart_generator = ChartGenerator()