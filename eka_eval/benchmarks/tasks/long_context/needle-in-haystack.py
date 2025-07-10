# eka_eval/benchmarks/tasks/long_context/needle_haystack.py

import torch
import sys
import argparse
import re
import json
import random
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import requests
import string
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEFAULT_CONTEXT_LENGTHS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
DEFAULT_DOCUMENT_DEPTH_PERCENTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
FINAL_CONTEXT_LENGTH_BUFFER = 200

# Needle configurations
DEFAULT_NEEDLE = "The special magic number mentioned by the research team is 42, which represents the ultimate answer to life."
DEFAULT_RETRIEVAL_QUESTION = "What is the special magic number mentioned by the research team?"
DEFAULT_EXPECTED_ANSWER = "42"

# Multi-needle setup for advanced testing
MULTI_NEEDLES = [
    {
        "text": "The secret password for the secure system is ALPHA7BRAVO9.",
        "question": "What is the secret password for the secure system?",
        "answer": "ALPHA7BRAVO9"
    },
    {
        "text": "The experiment was conducted on March 15, 2024 at precisely 14:30 UTC.",
        "question": "When was the experiment conducted?",
        "answer": "March 15, 2024 at precisely 14:30 UTC"
    },
    {
        "text": "The key research finding indicates that the optimal temperature is 273.15 Kelvin.",
        "question": "What is the optimal temperature according to the research finding?",
        "answer": "273.15 Kelvin"
    },
    {
        "text": "The project codename is Operation Phoenix and it involves advanced quantum computing.",
        "question": "What is the project codename?",
        "answer": "Operation Phoenix"
    }
]

class NeedleHaystackBenchmark:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.setup_model()
        self.haystack_content = self.load_haystack_content()
        
    def setup_model(self):
        """Initialize the model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Loading model {self.model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            logger.info("Creating text-generation pipeline...")
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            logger.info("Model setup complete.")
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def load_haystack_content(self) -> str:
        """Load or generate haystack content"""
        # Try to load Paul Graham essays (original benchmark content)
        paul_graham_url = "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/paul_graham_essay.txt"
        
        try:
            logger.info("Downloading Paul Graham essays for haystack content...")
            response = requests.get(paul_graham_url, timeout=30)
            response.raise_for_status()
            content = response.text
            logger.info(f"Loaded haystack content: {len(content)} characters")
            return content
        except Exception as e:
            logger.warning(f"Failed to download Paul Graham essays: {e}")
            logger.info("Generating synthetic haystack content...")
            return self.generate_synthetic_haystack()
    
    def generate_synthetic_haystack(self) -> str:
        """Generate synthetic haystack content as fallback"""
        # Create coherent, varied content that mimics real text
        topics = [
            "artificial intelligence and machine learning",
            "software development methodologies",
            "startup culture and entrepreneurship",
            "technology trends and innovation",
            "programming languages and frameworks",
            "data science and analytics",
            "computer science fundamentals",
            "business strategy and management",
            "scientific research and discovery",
            "education and learning systems"
        ]
        
        paragraphs = []
        for i in range(300):  # Generate substantial content
            topic = random.choice(topics)
            num_sentences = random.randint(4, 8)
            sentences = []
            
            for j in range(num_sentences):
                sentence_templates = [
                    f"Research in {topic} has shown significant advances in recent years.",
                    f"The implications of {topic} extend far beyond current applications.",
                    f"Experts in {topic} continue to debate the best approaches and methodologies.",
                    f"Recent developments in {topic} have opened new possibilities for innovation.",
                    f"The future of {topic} depends on continued collaboration and research.",
                    f"Students and professionals studying {topic} must understand both theory and practice."
                ]
                sentence = random.choice(sentence_templates)
                sentences.append(sentence)
            
            paragraph = " ".join(sentences)
            paragraphs.append(paragraph)
        
        haystack = "\n\n".join(paragraphs)
        logger.info(f"Generated synthetic haystack: {len(haystack)} characters")
        return haystack
    
    def get_context_length_in_tokens(self, text: str) -> int:
        """Get the actual token count for a text"""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def trim_context_to_token_length(self, text: str, target_length: int) -> str:
        """Trim text to fit within target token length"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= target_length:
            return text
        
        # Trim to target length
        trimmed_tokens = tokens[:target_length]
        return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
    
    def insert_needle_at_depth(self, haystack: str, needle: str, depth_percent: float) -> str:
        """Insert needle at specified depth percentage with precise positioning"""
        # Tokenize haystack
        haystack_tokens = self.tokenizer.encode(haystack, add_special_tokens=False)
        needle_tokens = self.tokenizer.encode(f" {needle} ", add_special_tokens=False)
        
        # Calculate precise insertion point
        insertion_point = int(len(haystack_tokens) * (depth_percent / 100.0))
        
        # Ensure insertion point is valid
        insertion_point = max(0, min(insertion_point, len(haystack_tokens)))
        
        # Insert needle tokens
        new_tokens = (
            haystack_tokens[:insertion_point] + 
            needle_tokens + 
            haystack_tokens[insertion_point:]
        )
        
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def create_test_context(self, context_length: int, depth_percent: float, 
                          needle_text: str) -> Tuple[str, int]:
        """Create a test context with needle inserted at specified depth"""
        # Calculate space needed for needle
        needle_tokens = len(self.tokenizer.encode(f" {needle_text} ", add_special_tokens=False))
        
        # Target length for haystack (accounting for needle and buffer)
        target_haystack_length = context_length - needle_tokens - FINAL_CONTEXT_LENGTH_BUFFER
        target_haystack_length = max(100, target_haystack_length)  # Minimum haystack size
        
        # Trim haystack to target length
        trimmed_haystack = self.trim_context_to_token_length(
            self.haystack_content, target_haystack_length
        )
        
        # Insert needle at specified depth
        context_with_needle = self.insert_needle_at_depth(
            trimmed_haystack, needle_text, depth_percent
        )
        
        # Get actual context length
        actual_length = self.get_context_length_in_tokens(context_with_needle)
        
        return context_with_needle, actual_length
    
    def format_prompt(self, context: str, question: str) -> str:
        """Format the prompt for the model with clear instructions"""
        return f"""You are a helpful assistant. Please read the following text carefully and answer the question based on the information provided.

<context>
{context}
</context>

Question: {question}

Please provide a direct and specific answer based only on the information in the context above.

Answer:"""
    
    def normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        
        # Remove common prefixes
        prefixes = [
            "the answer is", "answer:", "the", "a", "an",
            "according to the text", "based on the context",
            "the text states", "it is", "that is"
        ]
        
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def evaluate_response(self, response: str, expected_answer: str, needle_text: str) -> Dict[str, Any]:
        """Comprehensive evaluation of the response"""
        response_normalized = self.normalize_answer(response)
        expected_normalized = self.normalize_answer(expected_answer)
        needle_normalized = self.normalize_answer(needle_text)
        
        # Multiple evaluation criteria
        eval_results = {
            "raw_response": response,
            "normalized_response": response_normalized,
            "expected_answer": expected_answer,
            "normalized_expected": expected_normalized
        }
        
        # 1. Exact match with expected answer
        exact_match = expected_normalized in response_normalized
        
        # 2. Partial match with expected answer
        expected_tokens = set(expected_normalized.split())
        response_tokens = set(response_normalized.split())
        token_overlap = len(expected_tokens & response_tokens) / len(expected_tokens) if expected_tokens else 0
        
        # 3. Check if needle content appears in response
        needle_found = needle_normalized in response_normalized
        
        # 4. Fuzzy matching for numeric answers
        numeric_match = False
        if re.search(r'\d+', expected_answer):
            expected_nums = re.findall(r'\d+(?:\.\d+)?', expected_answer)
            response_nums = re.findall(r'\d+(?:\.\d+)?', response)
            numeric_match = any(num in response_nums for num in expected_nums)
        
        # Determine if retrieval was successful
        is_successful = (
            exact_match or 
            token_overlap >= 0.5 or 
            numeric_match or
            (needle_found and token_overlap > 0)
        )
        
        # Calculate confidence score
        confidence_score = max(
            1.0 if exact_match else 0.0,
            token_overlap,
            0.8 if numeric_match else 0.0,
            0.6 if needle_found else 0.0
        )
        
        eval_results.update({
            "exact_match": exact_match,
            "token_overlap": token_overlap,
            "needle_found": needle_found,
            "numeric_match": numeric_match,
            "is_successful": is_successful,
            "confidence_score": confidence_score
        })
        
        return eval_results
    
    def run_single_test(self, context_length: int, depth_percent: float, 
                       needle_config: Dict[str, str]) -> Dict[str, Any]:
        """Run a single needle-in-haystack test"""
        needle_text = needle_config["text"]
        question = needle_config["question"]
        expected_answer = needle_config["answer"]
        
        logger.debug(f"Running test: length={context_length}, depth={depth_percent}%")
        
        try:
            # Create test context
            test_context, actual_context_length = self.create_test_context(
                context_length, depth_percent, needle_text
            )
            
            # Format prompt
            prompt = self.format_prompt(test_context, question)
            
            # Generate response
            response = self.pipe(prompt)[0]['generated_text'].strip()
            
            # Evaluate response
            evaluation = self.evaluate_response(response, expected_answer, needle_text)
            
            # Compile results
            result = {
                "context_length": context_length,
                "actual_context_length": actual_context_length,
                "depth_percent": depth_percent,
                "needle_text": needle_text,
                "question": question,
                "expected_answer": expected_answer,
                "evaluation": evaluation,
                "success": evaluation["is_successful"],
                "score": evaluation["confidence_score"]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in single test: {e}")
            return {
                "context_length": context_length,
                "actual_context_length": 0,
                "depth_percent": depth_percent,
                "needle_text": needle_text,
                "question": question,
                "expected_answer": expected_answer,
                "error": str(e),
                "success": False,
                "score": 0.0
            }
    
    def run_benchmark(self, 
                     context_lengths: List[int] = None,
                     depth_percents: List[float] = None,
                     multi_needle: bool = False,
                     num_needles: int = 1) -> Dict[str, Any]:
        """Run the complete benchmark"""
        context_lengths = context_lengths or DEFAULT_CONTEXT_LENGTHS
        depth_percents = depth_percents or DEFAULT_DOCUMENT_DEPTH_PERCENTS
        
        logger.info("Starting Needle-in-a-Haystack benchmark...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Context lengths: {context_lengths}")
        logger.info(f"Depth percentages: {depth_percents}")
        
        # Select needles to test
        if multi_needle:
            needles_to_test = MULTI_NEEDLES[:num_needles]
            logger.info(f"Running multi-needle test with {len(needles_to_test)} needles")
        else:
            needles_to_test = [{
                "text": DEFAULT_NEEDLE,
                "question": DEFAULT_RETRIEVAL_QUESTION,
                "answer": DEFAULT_EXPECTED_ANSWER
            }]
            logger.info("Running single needle test")
        
        all_results = []
        total_tests = len(context_lengths) * len(depth_percents) * len(needles_to_test)
        
        with tqdm(total=total_tests, desc="Running tests") as pbar:
            for context_length in context_lengths:
                for depth_percent in depth_percents:
                    for needle_idx, needle_config in enumerate(needles_to_test):
                        result = self.run_single_test(
                            context_length, depth_percent, needle_config
                        )
                        result["needle_index"] = needle_idx
                        all_results.append(result)
                        
                        pbar.set_postfix({
                            'Length': context_length,
                            'Depth': f"{depth_percent}%",
                            'Success': result['success']
                        })
                        pbar.update(1)
        
        # Calculate summary statistics
        summary = self.calculate_summary_stats(all_results)
        
        benchmark_results = {
            "model_name": self.model_name,
            "benchmark_type": "multi_needle" if multi_needle else "single_needle",
            "num_needles": len(needles_to_test),
            "context_lengths": context_lengths,
            "depth_percents": depth_percents,
            "total_tests": len(all_results),
            "results": all_results,
            "summary": summary
        }
        
        return benchmark_results
    
    def calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results", "total_tests": len(results)}
        
        # Overall statistics
        success_rates = [r["success"] for r in valid_results]
        confidence_scores = [r["score"] for r in valid_results]
        
        # Group by context length
        by_context_length = {}
        for result in valid_results:
            length = result["context_length"]
            if length not in by_context_length:
                by_context_length[length] = {"success": [], "scores": []}
            by_context_length[length]["success"].append(result["success"])
            by_context_length[length]["scores"].append(result["score"])
        
        # Group by depth
        by_depth = {}
        for result in valid_results:
            depth = result["depth_percent"]
            if depth not in by_depth:
                by_depth[depth] = {"success": [], "scores": []}
            by_depth[depth]["success"].append(result["success"])
            by_depth[depth]["scores"].append(result["score"])
        
        summary = {
            "total_valid_tests": len(valid_results),
            "total_failed_tests": len(results) - len(valid_results),
            "overall_success_rate": np.mean(success_rates),
            "overall_average_score": np.mean(confidence_scores),
            "overall_median_score": np.median(confidence_scores),
            "perfect_retrieval_rate": np.mean([s == 1.0 for s in confidence_scores]),
            "by_context_length": {},
            "by_depth": {}
        }
        
        # Context length statistics
        for length, data in by_context_length.items():
            summary["by_context_length"][length] = {
                "success_rate": np.mean(data["success"]),
                "average_score": np.mean(data["scores"]),
                "median_score": np.median(data["scores"]),
                "num_tests": len(data["success"])
            }
        
        # Depth statistics
        for depth, data in by_depth.items():
            summary["by_depth"][depth] = {
                "success_rate": np.mean(data["success"]),
                "average_score": np.mean(data["scores"]),
                "median_score": np.median(data["scores"]),
                "num_tests": len(data["success"])
            }
        
        return summary


def evaluate_needle_haystack(
    pipe: Any,
    tokenizer: Any,
    model_name_for_logging: str,
    device: Any,
    context_lengths: List[int] = None,
    depth_percents: List[float] = None,
    multi_needle: bool = False,
    num_needles: int = 1,
    process_id: int = 0,
    gpu_id: int = 0,
    num_gpus: int = 1,
    results_dir: str = "results_output",
    save_outputs: bool = False,
    **kwargs
) -> Dict[str, float]:
    """Main evaluation function compatible with the evaluation framework"""
    
    # Create benchmark instance (reuse existing pipeline)
    benchmark = NeedleHaystackBenchmark.__new__(NeedleHaystackBenchmark)
    benchmark.model_name = model_name_for_logging
    benchmark.pipe = pipe
    benchmark.tokenizer = tokenizer
    benchmark.haystack_content = benchmark.load_haystack_content()
    
    # Set defaults
    context_lengths = context_lengths or [4000, 8000, 16000, 32000]  # Shorter for testing
    depth_percents = depth_percents or [0, 25, 50, 75, 100]  # Fewer points for testing
    
    logger.info(f"Starting Needle-in-Haystack evaluation: {model_name_for_logging}")
    
    # Run benchmark
    results = benchmark.run_benchmark(
        context_lengths=context_lengths,
        depth_percents=depth_percents,
        multi_needle=multi_needle,
        num_needles=num_needles
    )
    
    # Save detailed results if requested
    if save_outputs:
        os.makedirs(results_dir, exist_ok=True)
        model_safe = model_name_for_logging.replace('/', '_')
        benchmark_type = "multi_needle" if multi_needle else "single_needle"
        filename = f"needle_haystack_{benchmark_type}_{model_safe}_p{process_id}.json"
        output_path = os.path.join(results_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed results to {output_path}")
    
    # Extract summary metrics
    summary = results["summary"]
    
    logger.info(f"Needle-in-Haystack Results:")
    logger.info(f"  Overall Success Rate: {summary['overall_success_rate']:.1%}")
    logger.info(f"  Average Score: {summary['overall_average_score']:.3f}")
    logger.info(f"  Perfect Retrieval Rate: {summary['perfect_retrieval_rate']:.1%}")
    
    # Return metrics in standard format
    return {
        "NeedleHaystack": summary["overall_success_rate"] * 100,
        "NeedleHaystack_score": summary["overall_average_score"] * 100,
        "NeedleHaystack_perfect": summary["perfect_retrieval_rate"] * 100
    }


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    
    current_script_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))))
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
    
    from eka_eval.utils.logging_setup import setup_logging
    from eka_eval.core.model_loader import initialize_model_pipeline, cleanup_model_resources
    
    test_parser = argparse.ArgumentParser(description="Standalone Test Needle-in-Haystack")
    test_parser.add_argument("--model_name_test", type=str, default="meta-llama/Meta-Llama-3-8B")
    test_parser.add_argument("--multi_needle", action="store_true", help="Run multi-needle test")
    test_parser.add_argument("--num_needles", type=int, default=2, help="Number of needles for multi-needle test")
    test_parser.add_argument("--save_outputs", action="store_true", help="Save detailed outputs to JSON file")
    test_parser.add_argument("--context_lengths", nargs="+", type=int, default=[2000, 4000, 8000], help="Context lengths to test")
    test_parser.add_argument("--depth_percents", nargs="+", type=float, default=[0, 25, 50, 75, 100], help="Depth percentages to test")
    
    nh_args = test_parser.parse_args()
    setup_logging(level=logging.DEBUG, worker_id="NeedleHaystackTest")
    logger.info(f"--- Standalone Needle-in-Haystack Test: {nh_args.model_name_test} ---")
    
    nh_pipe, _ = initialize_model_pipeline(nh_args.model_name_test, target_device_id=0)
    if nh_pipe:
        nh_eval_args = {
            "pipe": nh_pipe,
            "tokenizer": nh_pipe.tokenizer,
            "model_name_for_logging": nh_args.model_name_test,
            "device": nh_pipe.device,
            "context_lengths": nh_args.context_lengths,
            "depth_percents": nh_args.depth_percents,
            "multi_needle": nh_args.multi_needle,
            "num_needles": nh_args.num_needles,
            "process_id": 0,
            "gpu_id": 0,
            "num_gpus": 1,
            "save_outputs": nh_args.save_outputs
        }
        try:
            print(json.dumps(evaluate_needle_haystack(**nh_eval_args), indent=2))
        finally:
            cleanup_model_resources(nh_pipe, getattr(nh_pipe, 'model', None))
    else:
        logger.error(f"Failed to init model {nh_args.model_name_test} for Needle-in-Haystack test.")