"""
Configuration settings for Engineer Performance Dashboard
"""
from enum import Enum

# App configuration
APP_TITLE = "🔧 Engineer Performance 360 Analytics"
APP_ICON = "📊"
APP_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Color schemes
COLOR_PALETTE = {
    'primary': '#1E3A8A',
    'secondary': '#2563EB',
    'success': '#10B981',
    'warning': '#FBBF24',
    'danger': '#EF4444',
    'info': '#3B82F6',
    'dark': '#1F2937',
    'light': '#F3F4F6',
    'gradient_blue': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_green': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    'gradient_orange': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient_cyan': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
}

# Status categories
class TaskStatus(Enum):
    DONE = ['Done', 'Closed', 'Completed', 'Resolved']
    IN_PROGRESS = ['In Progress', 'Progress', 'Working']
    PENDING = ['Pending', 'Unscheduled', 'Approval in Process']
    CANCELLED = ['Cancelled', 'Draft']
    
    @classmethod
    def categorize(cls, status):
        status_str = str(status).lower()
        for category, patterns in cls.__dict__.items():
            if not category.startswith('_') and isinstance(patterns, list):
                for pattern in patterns:
                    if pattern.lower() in status_str:
                        return category
        return 'OTHER'

# Priority levels
PRIORITY_ORDER = {
    '1-ASAP': 1,
    '2-High': 2,
    '3-Medium': 3,
    '4-Low': 4,
    'Not Specified': 5
}

# Date formats
DATE_FORMATS = [
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%m/%d/%Y %H:%M:%S',
    '%d-%m-%Y %H:%M:%S'
]

# File encodings to try
FILE_ENCODINGS = [
    'utf-8',
    'latin-1',
    'cp1252',
    'iso-8859-1',
    'utf-16',
    'utf-8-sig'
]

# Performance score weights
PERFORMANCE_WEIGHTS = {
    'completion_rate': 0.40,
    'efficiency': 0.30,
    'high_priority': 0.20,
    'accounts_served': 0.10
}