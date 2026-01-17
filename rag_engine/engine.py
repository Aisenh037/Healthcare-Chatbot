import importlib.util
from pathlib import Path
import sys

# Add the current directory to sys.path to ensure relative imports work inside the loaded modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def import_from_file(name, file_name):
    file_path = current_dir / file_name
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the class from the numbered file
rag_module = import_from_file("complete_rag", "6_complete_rag.py")
MinimalRAGChatbot = rag_module.MinimalRAGChatbot
