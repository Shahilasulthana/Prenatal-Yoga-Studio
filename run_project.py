# run_project.py - Place in your project root (cv_project folder)
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Mock TensorFlow to prevent import errors
import types

class MockModule(types.ModuleType):
    def __getattr__(self, name):
        return MockModule(f"{self.__name__}.{name}")
    
    def __call__(self, *args, **kwargs):
        return self

# Mock all tensorflow modules
sys.modules['tensorflow'] = MockModule('tensorflow')
sys.modules['tensorflow.tools'] = MockModule('tensorflow.tools')
sys.modules['tensorflow.tools.docs'] = MockModule('tensorflow.tools.docs')
sys.modules['tensorflow.python'] = MockModule('tensorflow.python')
sys.modules['tensorflow.python.framework'] = MockModule('tensorflow.python.framework')
sys.modules['tensorflow.core'] = MockModule('tensorflow.core')
sys.modules['tensorflow.core.framework'] = MockModule('tensorflow.core.framework')
sys.modules['tensorflow._api'] = MockModule('tensorflow._api')
sys.modules['tensorflow._api.v2'] = MockModule('tensorflow._api.v2')
sys.modules['tensorflow._api.v2.__internal__'] = MockModule('tensorflow._api.v2.__internal__')

print("✓ TensorFlow mocked successfully")

# Now import mediapipe
try:
    import mediapipe as mp
    print(f"✓ MediaPipe {mp.__version__} loaded")
except Exception as e:
    print(f"MediaPipe error: {e}")
    sys.exit(1)

# Import your main script
sys.path.insert(0, os.path.dirname(__file__))
from src.pose_extraction.angle_extractor import main

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Yoga Pose Angle Extractor")
    print("="*60 + "\n")
    main()