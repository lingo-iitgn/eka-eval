"""
Correct Needle-in-a-Haystack Benchmark Implementation
Based on Greg Kamradt's original work (2023)
GitHub: https://github.com/gkamradt/LLMTest_NeedleInAHaystack

This implementation fixes the issues in the provided code and follows the correct benchmark specification.
"""

import json
import random
import numpy as np
import os
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
import requests
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
RESULTS_VERSION = 1
NUM_CONCURRENT_REQUESTS = 1
SAVE_RESULTS = True
SAVE_CONTEXTS = False
FINAL_CONTEXT_LENGTH_BUFFER = 200
SECONDS_TO_SLEEP = None
PRINT_ONGOING_STATUS = True

# Context lengths to test (in tokens, not characters)
CONTEXT_LENGTHS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]

# Document depth percentages (where to place the needle)
DOCUMENT_DEPTH_PERCENTS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Multi-needle configuration
MULTI_NEEDLE = True
NUM_NEEDLES_TO_INSERT = 4
NUM_NEEDLES_TO_RETRIEVE = 2

# Default needle and question (can be customized)
DEFAULT_NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
DEFAULT_RETRIEVAL_QUESTION = "What is the best thing to do in San Francisco?"

# Multi-needle setup
MULTI_NEEDLES = [
    "The secret to happiness is eating pizza with pineapple on a Tuesday.",
    "The magic number for success is 42, discovered by deep thought computers.",
    "The best programming language is Python because it reads like English.",
    "The most important skill in life is the ability to learn continuously."
]

MULTI_NEEDLE_QUESTIONS = [
    "What is the secret to happiness?",
    "What is the magic number for success?", 
    "What is the best programming language?",
    "What is the most important skill in life?"
]

class NeedleHaystackBenchmark:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.setup_model()
        self.haystack_content = self.load_haystack_content()
        
    def setup_model(self):
        """Initialize the model and tokenizer"""
        print(f"Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto"
        )
        
        print("Creating text-generation pipeline...")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            max_new_tokens=100,
            do_sample=False,
            return_full_text=False
        )
        print("Model setup complete.")
    
    def load_haystack_content(self) -> str:
        """
        Load the Paul Graham essays as haystack content
        This is the original dataset used in Kamradt's benchmark
        """
        # Paul Graham essays URL (this is what the original benchmark uses)
        paul_graham_essays_url = "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/main/paul_graham_essay.txt"
        
        try:
            print("Downloading Paul Graham essays for haystack content...")
            response = requests.get(paul_graham_essays_url)
            response.raise_for_status()
            content = response.text
            print(f"Loaded haystack content: {len(content)} characters")
            return content
        except Exception as e:
            print(f"Failed to download Paul Graham essays: {e}")
            print("Using fallback synthetic content...")
            return self.generate_synthetic_haystack()
    
    def generate_synthetic_haystack(self) -> str:
        """Generate synthetic haystack content as fallback"""
        # Generate a large corpus of coherent text
        topics = [
            "artificial intelligence", "machine learning", "software development",
            "startup culture", "technology trends", "programming languages",
            "data science", "computer science", "innovation", "entrepreneurship"
        ]
        
        paragraphs = []
        for _ in range(200):  # Generate 200 paragraphs
            topic = random.choice(topics)
            sentences = []
            for _ in range(random.randint(3, 8)):  # 3-8 sentences per paragraph
                sentence = f"This is a sentence about {topic} and its impact on modern society."
                sentences.append(sentence)
            paragraphs.append(" ".join(sentences))
        
        return "\n\n".join(paragraphs)
    
    def get_context_length_in_tokens(self, text: str) -> int:
        """Get the actual token count for a text"""
        return len(self.tokenizer.encode(text))
    
    def trim_context_to_token_length(self, text: str, target_length: int) -> str:
        """Trim text to fit within target token length"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= target_length:
            return text
        
        # Trim to target length
        trimmed_tokens = tokens[:target_length]
        return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)
    
    def insert_needle_at_depth(self, haystack: str, needle: str, depth_percent: int) -> str:
        """Insert needle at specified depth percentage"""
        # Calculate insertion point
        haystack_tokens = self.tokenizer.encode(haystack)
        insertion_point = int(len(haystack_tokens) * (depth_percent / 100))
        
        # Insert needle
        needle_tokens = self.tokenizer.encode(needle, add_special_tokens=False)
        
        # Combine tokens
        new_tokens = (
            haystack_tokens[:insertion_point] + 
            needle_tokens + 
            haystack_tokens[insertion_point:]
        )
        
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def insert_multiple_needles(self, haystack: str, needles: List[str], depth_percent: int) -> str:
        """Insert multiple needles with even spacing after the first needle"""
        if not needles:
            return haystack
        
        # Insert first needle at specified depth
        context_with_needles = self.insert_needle_at_depth(haystack, needles[0], depth_percent)
        
        if len(needles) == 1:
            return context_with_needles
        
        # Calculate spacing for remaining needles
        remaining_space = 100 - depth_percent
        depth_interval = remaining_space / len(needles)
        
        # Insert remaining needles
        for i, needle in enumerate(needles[1:], 1):
            needle_depth = depth_percent + (i * depth_interval)
            context_with_needles = self.insert_needle_at_depth(
                context_with_needles, needle, needle_depth
            )
        
        return context_with_needles
    
    def create_test_context(self, context_length: int, depth_percent: int, 
                          needles: List[str]) -> str:
        """Create a test context with needles inserted"""
        # Trim haystack to approximate target length (accounting for needles)
        needle_tokens = sum(len(self.tokenizer.encode(needle)) for needle in needles)
        target_haystack_length = context_length - needle_tokens - FINAL_CONTEXT_LENGTH_BUFFER
        
        trimmed_haystack = self.trim_context_to_token_length(
            self.haystack_content, target_haystack_length
        )
        
        # Insert needles
        if len(needles) == 1:
            return self.insert_needle_at_depth(trimmed_haystack, needles[0], depth_percent)
        else:
            return self.insert_multiple_needles(trimmed_haystack, needles, depth_percent)
    
    def format_prompt(self, context: str, question: str) -> str:
        """Format the prompt for the model"""
        return f"""You are a helpful AI assistant. Please answer the question based on the context provided.

Context: {context}

Question: {question}

Answer:"""
    
    def evaluate_response(self, response: str, expected_answer: str) -> bool:
        """Evaluate if the response contains the expected answer"""
        # Simple containment check (can be made more sophisticated)
        response_lower = response.lower().strip()
        expected_lower = expected_answer.lower().strip()
        
        # Check if key parts of the expected answer are in the response
        return expected_lower in response_lower
    
    def run_single_test(self, context_length: int, depth_percent: int, 
                       needles: List[str], questions: List[str]) -> Dict[str, Any]:
        """Run a single needle-in-haystack test"""
        # Create test context
        test_context = self.create_test_context(context_length, depth_percent, needles)
        actual_context_length = self.get_context_length_in_tokens(test_context)
        
        results = {
            "context_length": context_length,
            "actual_context_length": actual_context_length,
            "depth_percent": depth_percent,
            "needles": needles,
            "questions": questions,
            "individual_results": [],
            "retrieval_scores": []
        }
        
        # Test each needle/question pair
        for i, (needle, question) in enumerate(zip(needles, questions)):
            prompt = self.format_prompt(test_context, question)
            
            try:
                # Generate response
                response = self.pipe(prompt)[0]['generated_text'].strip()
                
                # Evaluate response
                is_correct = self.evaluate_response(response, needle)
                score = 1.0 if is_correct else 0.0
                
                individual_result = {
                    "needle_index": i,
                    "needle": needle,
                    "question": question,
                    "response": response,
                    "is_correct": is_correct,
                    "score": score
                }
                
                results["individual_results"].append(individual_result)
                results["retrieval_scores"].append(score)
                
            except Exception as e:
                print(f"Error processing needle {i}: {e}")
                individual_result = {
                    "needle_index": i,
                    "needle": needle,
                    "question": question,
                    "response": f"ERROR: {e}",
                    "is_correct": False,
                    "score": 0.0
                }
                results["individual_results"].append(individual_result)
                results["retrieval_scores"].append(0.0)
        
        # Calculate overall scores
        results["average_score"] = np.mean(results["retrieval_scores"])
        results["num_correct"] = sum(results["retrieval_scores"])
        results["total_needles"] = len(needles)
        
        return results
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark"""
        print("Starting Needle-in-a-Haystack benchmark...")
        
        all_results = []
        
        # Determine needles and questions to use
        if MULTI_NEEDLE:
            needles = MULTI_NEEDLES[:NUM_NEEDLES_TO_INSERT]
            questions = MULTI_NEEDLE_QUESTIONS[:NUM_NEEDLES_TO_INSERT]
            print(f"Running multi-needle test with {len(needles)} needles")
        else:
            needles = [DEFAULT_NEEDLE]
            questions = [DEFAULT_RETRIEVAL_QUESTION]
            print("Running single needle test")
        
        # Run tests for all combinations
        total_tests = len(CONTEXT_LENGTHS) * len(DOCUMENT_DEPTH_PERCENTS)
        
        with tqdm(total=total_tests, desc="Running tests") as pbar:
            for context_length in CONTEXT_LENGTHS:
                for depth_percent in DOCUMENT_DEPTH_PERCENTS:
                    if PRINT_ONGOING_STATUS:
                        print(f"\nTesting context length: {context_length}, depth: {depth_percent}%")
                    
                    try:
                        result = self.run_single_test(
                            context_length, depth_percent, needles, questions
                        )
                        all_results.append(result)
                        
                        if PRINT_ONGOING_STATUS:
                            print(f"Score: {result['average_score']:.2f}")
                            
                    except Exception as e:
                        print(f"Error in test (length={context_length}, depth={depth_percent}%): {e}")
                        # Add error result
                        error_result = {
                            "context_length": context_length,
                            "depth_percent": depth_percent,
                            "error": str(e),
                            "average_score": 0.0,
                            "num_correct": 0,
                            "total_needles": len(needles)
                        }
                        all_results.append(error_result)
                    
                    pbar.update(1)
        
        # Compile final results
        benchmark_results = {
            "model_name": self.model_name,
            "benchmark_type": "multi_needle" if MULTI_NEEDLE else "single_needle",
            "num_needles": len(needles),
            "context_lengths": CONTEXT_LENGTHS,
            "depth_percents": DOCUMENT_DEPTH_PERCENTS,
            "results": all_results,
            "summary": self.calculate_summary_stats(all_results)
        }
        
        return benchmark_results
    
    def calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        scores = [r["average_score"] for r in valid_results]
        
        # Group by context length
        by_context_length = {}
        for result in valid_results:
            length = result["context_length"]
            if length not in by_context_length:
                by_context_length[length] = []
            by_context_length[length].append(result["average_score"])
        
        # Group by depth
        by_depth = {}
        for result in valid_results:
            depth = result["depth_percent"]
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(result["average_score"])
        
        summary = {
            "overall_average_score": np.mean(scores),
            "overall_median_score": np.median(scores),
            "overall_std": np.std(scores),
            "perfect_retrieval_rate": np.mean([s == 1.0 for s in scores]),
            "by_context_length": {
                length: {
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores)
                }
                for length, scores in by_context_length.items()
            },
            "by_depth": {
                depth: {
                    "mean": np.mean(scores),
                    "median": np.median(scores),
                    "std": np.std(scores)
                }
                for depth, scores in by_depth.items()
            }
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file"""
        if not SAVE_RESULTS:
            return
        
        os.makedirs("results", exist_ok=True)
        
        model_name_clean = self.model_name.replace("/", "_")
        benchmark_type = "multi_needle" if MULTI_NEEDLE else "single_needle"
        filename = f"results/needle_haystack_{model_name_clean}_{benchmark_type}_v{RESULTS_VERSION}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of results"""
        print("\n" + "="*60)
        print("NEEDLE-IN-A-HAYSTACK BENCHMARK RESULTS")
        print("="*60)
        
        summary = results["summary"]
        
        print(f"Model: {results['model_name']}")
        print(f"Benchmark Type: {results['benchmark_type']}")
        print(f"Number of Needles: {results['num_needles']}")
        print(f"Context Lengths Tested: {results['context_lengths']}")
        print(f"Depth Percentages Tested: {results['depth_percents']}")
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Average Score: {summary['overall_average_score']:.1%}")
        print(f"  Median Score: {summary['overall_median_score']:.1%}")
        print(f"  Perfect Retrieval Rate: {summary['perfect_retrieval_rate']:.1%}")
        
        print(f"\nPERFORMANCE BY CONTEXT LENGTH:")
        for length, stats in summary['by_context_length'].items():
            print(f"  {length:>6} tokens: {stats['mean']:.1%} (±{stats['std']:.1%})")
        
        print(f"\nPERFORMANCE BY DEPTH:")
        for depth, stats in summary['by_depth'].items():
            print(f"  {depth:>3}% depth: {stats['mean']:.1%} (±{stats['std']:.1%})")


def main():
    """Main execution function"""
    print("Initializing Needle-in-a-Haystack Benchmark...")
    
    benchmark = NeedleHaystackBenchmark(MODEL_NAME)
    results = benchmark.run_benchmark()
    
    benchmark.print_summary(results)
    benchmark.save_results(results)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
main()