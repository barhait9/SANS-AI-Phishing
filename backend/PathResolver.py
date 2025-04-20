# path_utils.py
import os

class PathResolver:
    """Centralized path resolution for the entire application"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.environ.get("DATA_DIR", os.path.join(self.base_dir, "DataHandle"))
    
    def resolve(self, filename):
        """Resolve file paths with multiple fallback locations"""
        # Tries adjacent to source files first
        local_path = os.path.join(self.base_dir, filename)
        if os.path.exists(local_path):
            return local_path
            
        # Tries data subdir if not adjacent
        data_path = os.path.join(self.data_dir, filename)
        if os.path.exists(data_path):
            return data_path
            
        # Tries direct path if not adjacent or same subdir
        if os.path.exists(filename):
            return os.path.abspath(filename)
            
        raise FileNotFoundError(
            f"Could not find {filename} in: {[local_path, data_path, filename]}"
            f"\nCurrent working directory: {os.getcwd()}"
        )
    def printValues(self):
        print("base_dir:" + self.base_dir)
        print("data dir:" + self.data_dir)


path_resolver = PathResolver()
