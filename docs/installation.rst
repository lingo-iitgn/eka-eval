============
Installation
============

System Requirements
===================

* Python 3.8+
* PyTorch 2.0+
* CUDA (optional, for GPU support)
* 8GB+ RAM (16GB+ recommended for large models)

1. Clone the Repository
=======================

.. code-block:: bash

   git clone https://github.com/lingo-iitgn/eka-eval.git
   cd eka-eval

2. Environment Setup
====================

.. code-block:: bash

   # Create virtual environment
   python3 -m venv eka-env
   source eka-env/bin/activate  # On Windows: eka-env\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

3. Required Dependencies
========================

The following packages are required:

.. code-block:: text

   torch>=2.0.0
   transformers>=4.35.0
   datasets>=2.14.0
   evaluate>=0.4.0
   accelerate>=0.24.0
   bitsandbytes>=0.41.0  # For quantization
   pandas>=1.5.0
   tqdm>=4.64.0
   numpy>=1.24.0

4. Authentication (Optional)
============================

For private models or gated datasets:

.. code-block:: bash

   huggingface-cli login
   # OR
   export HF_TOKEN="your_hf_token_here"

Verification
============

Test your installation:

.. code-block:: python

   import eka_eval
   print("Eka-Eval installed successfully!")

.. code-block:: bash

   # Test with a simple evaluation
   python3 scripts/run_benchmarks.py --help