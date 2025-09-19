import tiktoken
import time
from resource_monitor import ResourceMonitor, monitor_resources, profile_memory

class TokenUsageTracker:
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.total_tokens = 0
        self.total_api_calls = 0
        self.start_time = time.time()

    @monitor_resources
    def count_tokens(self, text, model="gpt-3.5-turbo"):
        """
        Count the number of tokens in a text string.
        While this uses GPT tokenizer, it gives a reasonable approximation for Gemini
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            token_count = len(encoding.encode(text))
            self.total_tokens += token_count
            self.total_api_calls += 1
            self._log_usage()
            return token_count
        except Exception as e:
            print(f"Error counting tokens: {e}")
            # Fallback to rough estimation (not as accurate)
            estimated_tokens = int(len(text.split()) * 1.3)  # Rough estimate
            self.total_tokens += estimated_tokens
            self.total_api_calls += 1
            self._log_usage()
            return estimated_tokens

    def _log_usage(self):
        """Log token usage statistics."""
        elapsed_time = time.time() - self.start_time
        stats = self.resource_monitor.get_system_stats()
        
        usage_info = (
            f"\nToken Usage Statistics:\n"
            f"Total Tokens: {self.total_tokens}\n"
            f"Total API Calls: {self.total_api_calls}\n"
            f"Average Tokens per Call: {self.total_tokens/max(1, self.total_api_calls):.2f}\n"
            f"Tokens per Second: {self.total_tokens/max(1, elapsed_time):.2f}\n"
            f"System Resource Usage:\n"
            f"  CPU: {stats['cpu_percent']}%\n"
            f"  Memory: {stats['memory_percent']}%\n"
            f"  Process Memory: {stats['process_memory']:.2f} MB\n"
        )
        
        with open("logs/token_usage.log", "a") as f:
            f.write(usage_info)
            
    def get_usage_stats(self):
        """Get current usage statistics."""
        return {
            'total_tokens': self.total_tokens,
            'total_api_calls': self.total_api_calls,
            'average_tokens_per_call': self.total_tokens/max(1, self.total_api_calls),
            'tokens_per_second': self.total_tokens/max(1, time.time() - self.start_time),
            'system_stats': self.resource_monitor.get_system_stats()
        }

# Create a global instance for easy access
tracker = TokenUsageTracker()
