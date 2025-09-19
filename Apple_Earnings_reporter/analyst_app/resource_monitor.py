import psutil
import os
import time
from memory_profiler import profile
from functools import wraps
from datetime import datetime

class ResourceMonitor:
    def __init__(self, log_dir="logs"):
        """Initialize the resource monitor with an optional log directory."""
        self.log_dir = log_dir
        self.process = psutil.Process(os.getpid())
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up the logging directory if it doesn't exist."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def get_system_stats(self):
        """Get current system resource statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'process_memory': self.process.memory_info().rss / 1024 / 1024,  # MB
        }
        
    def log_resources(self, operation_name=""):
        """Log current resource usage."""
        stats = self.get_system_stats()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = (f"{timestamp} - {operation_name}\n"
                    f"CPU Usage: {stats['cpu_percent']}%\n"
                    f"Memory Usage: {stats['memory_percent']}%\n"
                    f"Disk Usage: {stats['disk_usage']}%\n"
                    f"Process Memory: {stats['process_memory']:.2f} MB\n"
                    f"{'='*50}\n")
        
        log_file = os.path.join(self.log_dir, 'resource_usage.log')
        with open(log_file, 'a') as f:
            f.write(log_entry)
        return stats

def monitor_resources(func):
    """Decorator to monitor resource usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        start_time = time.time()
        monitor.log_resources(f"Starting {func.__name__}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        monitor.log_resources(f"Finished {func.__name__} (took {execution_time:.2f}s)")
        
        return result
    return wrapper

# Memory profiling decorator
def profile_memory(func):
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    @profile
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
