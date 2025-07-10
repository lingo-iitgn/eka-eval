import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm 
import json 
from datetime import datetime
from collections import defaultdict
import evaluate as hf_evaluate
import gc
from typing import List, Dict
import argparse


BENCHMARK_CONFIG = {
"CODE GENERATION": {
"HumanEval": {"description": "Average pass@1 scores on HumanEval.", "evaluation_function": "evaluate_humaneval"},
"MBPP": {"description": "Average pass@1 scores on MBPP.", "evaluation_function": "evaluate_mbpp"}
},
"READING COMPREHENSION": {
"SQuAD": {"description": "0-shot average", "evaluation_function": "evaluate_squad"},
"QuAC": {"description": "0-shot average", "evaluation_function": "evaluate_quac"},
"BoolQ": {"description": "0-shot average", "evaluation_function": "evaluate_boolq"},
},
"MATH EVAL": {
"GSM8K": {"description": "8-shot average", "evaluation_function": "evaluate_gsm8k"},
"MATH": {"description":"4-shot average","evaluation_function":"evaluate_math"}
},
"COMMONSENSE REASONING": { "PIQA": {"description": "0-shot", "evaluation_function": "evaluate_commonsense_reasoning"}},
"WORLD KNOWLEDGE": {"NaturalQuestions": {"description": "5-shot average", "evaluation_function": "evaluate_world_knowledge"}},
"MMLU": {"description": "5-shot overall", "evaluation_function": "evaluate_mmlu"},
"BBH": {"description": "3-shot overall", "evaluation_function": "evaluate_bbh"},
"AGIEval": {"description": "3-5 shot overall", "evaluation_function": "evaluate_agieval"},}

try:
    from eval_tasks.code.humaneval import evaluate_humaneval 
except ImportError:
    print("WARN main.py: Could not import evaluate_humaneval. Using placeholder.")
    def evaluate_humaneval(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"HumanEval": pd.NA}
try:
    from eval_tasks.code.mbpp import evaluate_mbpp
except ImportError:
    print("WARN main.py: Could not import evaluate_mbpp. Using placeholder.")
    def evaluate_mbpp(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"MBPP": pd.NA}
try:
    from eval_tasks.math.gsm8k import evaluate_gsm8k # Corrected path if gsm8k.py is in eval_tasks/math/
                                                   # Or from eval_tasks.gsm8k import evaluate_gsm8k if in eval_tasks/
except ImportError:
    print("WARN main.py: Could not import evaluate_gsm8k. Using placeholder.")
    def evaluate_gsm8k(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"GSM8K": pd.NA}
try:
    from eval_tasks.math.math import evaluate_math # Corrected path if math.py is in eval_tasks/math/
except ImportError:
    print("WARN main.py: Could not import evaluate_math (for Hendrycks). Using placeholder.")
    def evaluate_math(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"MATH": pd.NA}
try:
    from eval_tasks.reading_comprehension.boolq import evaluate_boolq 
except ImportError:
    print("WARN main.py: Could not import evaluate_boolq. Using placeholder.")
    def evaluate_boolq(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"BoolQ": pd.NA}
try:
    from eval_tasks.reading_comprehension.squad import evaluate_squad
except ImportError:
    print("WARN main.py: Could not import evaluate_squad. Using placeholder.")
    def evaluate_squad(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"SQuAD": pd.NA}
try:
    from eval_tasks.reading_comprehension.quac import evaluate_quac
except ImportError:
    print("WARN main.py: Could not import evaluate_quac. Using placeholder.")
    def evaluate_quac(model_name: str, pipe, model_size_gb: float, **kwargs) -> Dict[str, float]: return {"QuAC": pd.NA}


PRE_CALCULATED_RESULTS = pd.DataFrame()

# --- Placeholder functions (for tasks not yet fully implemented in separate files) ---
def evaluate_commonsense_reasoning_placeholder(model_name: str, pipe, model_size_gb: float, benchmark: str, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} for {model_name} (Placeholder from main.py) ---")
    return {benchmark: 60.0 + (hash(model_name + benchmark) % 10)}

def evaluate_world_knowledge_placeholder(model_name: str, pipe, model_size_gb: float, benchmark: str, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} for {model_name} (Placeholder from main.py) ---")
    return {benchmark: 75.0 + (hash(model_name + benchmark) % 10)}

def evaluate_reading_comprehension_placeholder(model_name: str, pipe, model_size_gb: float, benchmark: str, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} for Reading Comp (Placeholder from main.py) ---")
    return {benchmark: 80.1 + (hash(model_name + benchmark) % 10)}

def evaluate_math_placeholder(model_name: str, pipe, model_size_gb: float, benchmark: str, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running {benchmark} for MATH category (Placeholder from main.py) ---")
    return {benchmark: 15.0 + (hash(model_name + benchmark) % 5)}

def evaluate_mmlu_placeholder(model_name: str, pipe, model_size_gb: float, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running MMLU for {model_name} (Placeholder from main.py) ---")
    return {"MMLU": 65.0 + (hash(model_name) % 15)}

def evaluate_bbh_placeholder(model_name: str, pipe, model_size_gb: float, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running BBH for {model_name} (Placeholder from main.py) ---")
    return {"BBH": 50.0 + (hash(model_name) % 10)}

def evaluate_agieval_placeholder(model_name: str, pipe, model_size_gb: float, batch_size: int = 1, **kwargs) -> Dict[str, float]:
    print(f"\n--- Running AGIEval for {model_name} (Placeholder from main.py) ---")
    return {"AGIEval": 40.0 + (hash(model_name) % 10)}

def initialize_pipeline(model_name: str, device_id: int = 0):
    device_map_arg = {'': device_id} if torch.cuda.is_available() else "cpu"
    target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    num_added_tokens = 0 
    try: tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
    except Exception as e: print(f"Tokenizer load fail for {model_name}: {e}"); return None, 'N/A'
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens_to_add = ["[END]"] if "[END]" not in tokenizer.get_vocab() else [] # Simpler check
    if special_tokens_to_add:
        num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_to_add})
        if num_added_tokens > 0: print(f"Added {num_added_tokens} special token(s): {special_tokens_to_add}")
    
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=target_dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True) if torch.cuda.is_available() else None
    model = None
    try:
        print(f"Loading model {model_name} {'with 4-bit quant' if quant_config else ('on CPU' if not torch.cuda.is_available() else '')}...")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device_map_arg, quantization_config=quant_config, torch_dtype=target_dtype, attn_implementation="eager", low_cpu_mem_usage=True)
        print(f"Model {model_name} loaded.")
    except Exception as e: 
        print(f"Warn: Initial model load failed for {model_name}: {e}. Trying without explicit quant_config if on GPU.")
        if torch.cuda.is_available() and quant_config is not None: 
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map=device_map_arg, torch_dtype=target_dtype, attn_implementation="eager", low_cpu_mem_usage=True)
                print(f"Model {model_name} loaded without explicit quantization_config.")
            except Exception as e2: print(f"FATAL: Model load failed for {model_name} (attempt 2): {e2}"); return None, 'N/A'
        else:
            print(f"FATAL: Model load failed for {model_name}: {e}"); return None, 'N/A'

    if model and num_added_tokens > 0: model.resize_token_embeddings(len(tokenizer)); print(f"Resized model embeddings to {len(tokenizer)}.")
    
    param_count_str = 'N/A'
    if model:
        try:
            total_params = sum(p.numel() for p in model.parameters())
            if "gemma-2b" in model_name.lower() or "gemma_2b" in model_name.lower() : param_count_str = "2.00" # Override for Gemma 2B
            elif "llama-7b" in model_name.lower() : param_count_str = "7.00"
            elif total_params > 0: param_count_str = f"{total_params / 1_000_000_000:.2f}"
            else: param_count_str = "0.00"
        except Exception as e: print(f"Warn: Calc param count failed: {e}")
    
    try: pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=target_dtype)
    except Exception as e: print(f"Pipeline creation failed for {model_name}: {e}"); return None, param_count_str
    return pipe, param_count_str


def run_evaluation_for_model_and_tasks(
    input_model_name: str,
    task_group_to_evaluate: str, 
    selected_benchmarks_for_group: List[str], 
    process_id: int,
    physical_gpu_id: int,
    num_total_workers: int, 
    orchestrator_batch_size: int 
):
    global PRE_CALCULATED_RESULTS
    model_name_lower = input_model_name.lower()
    csv_file_path = 'calculated.csv'
    
    if os.path.exists(csv_file_path):
        try:
            temp_df = pd.read_csv(csv_file_path)
            if 'Model' in temp_df.columns: PRE_CALCULATED_RESULTS = temp_df[temp_df['Model'].str.lower() == model_name_lower].copy()
            else: PRE_CALCULATED_RESULTS = pd.DataFrame()
        except Exception as e: print(f"P{process_id} Warn: Load pre-calc fail: {e}"); PRE_CALCULATED_RESULTS = pd.DataFrame()
    else: PRE_CALCULATED_RESULTS = pd.DataFrame()

    model_pipeline, model_param_count_str = initialize_pipeline(input_model_name, device_id=0) 
    if model_pipeline is None: print(f"P{process_id}: Fail init model {input_model_name}. Skip."); return
    
    print(f"P{process_id} (GPU{physical_gpu_id}): Init {input_model_name} ({model_param_count_str}B params). OrchBS:{orchestrator_batch_size}")
    print(f"P{process_id}: Eval TG '{task_group_to_evaluate}' for BMs: {selected_benchmarks_for_group}")

    new_results = [] 
    task_group_name = task_group_to_evaluate 
    if task_group_name not in BENCHMARK_CONFIG: print(f"P{process_id}: TG '{task_group_name}' !in BENCHMARK_CONFIG. Skip."); return
        
    task_cfg_grp = BENCHMARK_CONFIG[task_group_name]
    is_single_bm_grp = task_group_name in ["MMLU", "BBH", "AGIEval"] 
    scores_for_avg = [] 

    for bm_name in selected_benchmarks_for_group: 
        current_score = pd.NA 
        if not PRE_CALCULATED_RESULTS.empty:
            pre_calc = PRE_CALCULATED_RESULTS[(PRE_CALCULATED_RESULTS['Task'].str.lower() == task_group_name.lower()) & (PRE_CALCULATED_RESULTS['Benchmark'].str.lower() == bm_name.lower()) & (PRE_CALCULATED_RESULTS['Score'].notna())]
            if not pre_calc.empty:
                current_score = pre_calc['Score'].iloc[0]
                print(f"P{process_id}: Pre-calc for {task_group_name}-{bm_name}: {current_score}")
                if pd.notna(current_score): scores_for_avg.append(float(current_score))
                new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':bm_name,'Score':current_score,'Timestamp':datetime.now().isoformat()})
                continue 
        
        eval_fn_name = task_cfg_grp.get("evaluation_function") if is_single_bm_grp else task_cfg_grp.get(bm_name, {}).get("evaluation_function")
        if not eval_fn_name: print(f"P{process_id}: No eval_fn for {task_group_name}-{bm_name}."); new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':bm_name,'Score':pd.NA,'Timestamp':datetime.now().isoformat()}); continue
        eval_fn = globals().get(eval_fn_name)
        if not eval_fn: print(f"P{process_id}: Eval_fn '{eval_fn_name}' not found for {bm_name}."); new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':bm_name,'Score':pd.NA,'Timestamp':datetime.now().isoformat()}); continue
        
        try:
            print(f"P{process_id}: Eval {task_group_name}-{bm_name} via '{eval_fn_name}'...")
            args_eval = {
                "model_name":input_model_name, "pipe":model_pipeline, 
                "model_size_gb":0.0, 
                "batch_size":orchestrator_batch_size 
            }
            generic_eval_functions_expecting_benchmark_kwarg = [
                "evaluate_commonsense_reasoning_placeholder", 
                "evaluate_world_knowledge_placeholder",
                "evaluate_reading_comprehension_placeholder",
                "evaluate_math_placeholder" 
            ]
            if eval_fn_name in generic_eval_functions_expecting_benchmark_kwarg: 
                args_eval["benchmark"] = bm_name 
            if eval_fn_name == "evaluate_mbpp": 
                args_eval.update({"num_samples_per_task":1, "batch_size":1, "dataset_split":"test[:5]", "include_tests_in_prompt":False})
            elif eval_fn_name == "evaluate_humaneval": 
                args_eval.update({"num_samples_per_task":1, "dataset_split":"test", "k_values":[1], "use_fewshot":True, "batch_size":1})
            elif eval_fn_name == "evaluate_boolq":
                cp_dir = os.path.join("checkpoints","boolq_checkpoints"); os.makedirs(cp_dir, exist_ok=True)
                args_eval.update({"batch_size":8, "dataset_split":"validation[:20]", "process_id":process_id, "gpu_id":physical_gpu_id, "num_gpus":num_total_workers, "checkpoint_dir":cp_dir, "resume":True})
            elif eval_fn_name == "evaluate_squad":
                cp_dir_squad = os.path.join("checkpoints", "squad_checkpoints"); os.makedirs(cp_dir_squad, exist_ok=True)
                args_eval.update({"batch_size":4, "dataset_split":"validation[:10]", "process_id":process_id, "gpu_id":physical_gpu_id, "num_gpus":num_total_workers, "checkpoint_dir":cp_dir_squad, "resume":True})
            elif eval_fn_name == "evaluate_quac": # Added QuAC
                cp_dir_quac = os.path.join("checkpoints", "quac_checkpoints"); os.makedirs(cp_dir_quac, exist_ok=True)
                args_eval.update({"batch_size":4, "dataset_split":"validation[:10]", "process_id":process_id, "gpu_id":physical_gpu_id, "num_gpus":num_total_workers, "checkpoint_dir":cp_dir_quac, "resume":True})
            elif eval_fn_name == "evaluate_gsm8k":
                args_eval["batch_size"] = 4
                args_eval["dataset_split"] = "test[:10]" 
            elif eval_fn_name == "evaluate_math": 
                args_eval["batch_size"] = 4 
                args_eval["dataset_split"] = "test[:10]"
                args_eval["process_id"] = process_id
                args_eval["gpu_id"] = physical_gpu_id
                args_eval["num_gpus"] = num_total_workers
                math_cp_dir = os.path.join("checkpoints", "hendrycks_math_checkpoints"); os.makedirs(math_cp_dir, exist_ok=True)
                args_eval["checkpoint_dir"] = math_cp_dir
                args_eval["resume"] = True
            
            scores_dict_from_eval = {} 
            try:
                print(f"DEBUG main.py: Calling {eval_fn_name} with arg keys: {list(args_eval.keys())}")
                scores_dict_from_eval = eval_fn(**args_eval) 
            except TypeError as te:
                functions_with_own_batch_handling = ["evaluate_mbpp", "evaluate_humaneval", "evaluate_boolq", "evaluate_squad", "evaluate_quac", "evaluate_gsm8k", "evaluate_math"]
                should_retry = "batch_size" in str(te) and eval_fn_name not in functions_with_own_batch_handling
                if should_retry:
                    print(f"Warn: Eval func {eval_fn_name} no 'batch_size'. Retry."); args_eval.pop("batch_size")
                    scores_dict_from_eval = eval_fn(**args_eval) 
                else: 
                    print(f"DEBUG main.py: TypeError for {eval_fn_name} not handled by retry. Error: {te}. Args used: {args_eval}")
                    raise te 
            current_score = scores_dict_from_eval.get(bm_name, pd.NA) 
            new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':bm_name,'Score':current_score if pd.notna(current_score) else pd.NA,'Timestamp':datetime.now().isoformat()})
            if pd.notna(current_score): scores_for_avg.append(float(current_score))
            else: print(f"Warn: No valid score for {task_group_name}-{bm_name} from '{eval_fn_name}'. Dict: {scores_dict_from_eval}")
        except Exception as e: 
            print(f"\033[91mP{process_id}: Outer error during evaluation of {task_group_name}-{bm_name}: {e}\033[0m"); 
            import traceback; traceback.print_exc()
            new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':bm_name,'Score':pd.NA,'Timestamp':datetime.now().isoformat()})
    
    if not is_single_bm_grp and len(scores_for_avg) > 1:
        avg = sum(scores_for_avg)/len(scores_for_avg); print(f"P{process_id}: Avg for {task_group_name} ({len(scores_for_avg)} scores): {avg:.2f}")
        new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':'Average','Score':avg,'Timestamp':datetime.now().isoformat()})
    elif not is_single_bm_grp and scores_for_avg: 
        print(f"P{process_id}: Only one score ({scores_for_avg[0]:.2f}) for {task_group_name}. No 'Average' stored.")

    # Save all new results
    if new_results:
        new_df = pd.DataFrame(new_results)
        try:
            if os.path.exists(csv_file_path):
                existing_df = pd.read_csv(csv_file_path); 
                combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['Model','Task','Benchmark'],keep='last')
            else: combined_df = new_df
            combined_df.to_csv(csv_file_path,index=False); print(f"P{process_id}: Results saved to '{csv_file_path}'.")
        except Exception as e: print(f"\033[91mP{process_id}: Error save CSV: {e}\033[0m")

    # Model cleanup
    if model_pipeline:
        try:
            if hasattr(model_pipeline,'model') and model_pipeline.model: del model_pipeline.model
            del model_pipeline; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache(); 
            print(f"P{process_id}: Model cleaned up.")
        except Exception as e: print(f"P{process_id}: Error model cleanup: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark Worker (main.py).")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1, help="Total workers (for logging/splitting).")
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_group", type=str, required=True, help="Single task group for this worker.")
    parser.add_argument("--selected_benchmarks_json", type=str, required=True, help="JSON map task_group -> list_of_bms.")
    parser.add_argument("--batch_size", type=int, default=1, help="Orchestrator's default batch_size.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"Proc {args.process_id}: Set CUDA_VISIBLE_DEVICES={args.gpu_id}. Using logical cuda:0.")
    else: print(f"Proc {args.process_id}: CUDA not available. CPU.")

    try: selected_benchmarks_map = json.loads(args.selected_benchmarks_json)
    except json.JSONDecodeError as e: print(f"P{args.process_id}: FATAL Decode JSON: {e}. JSON: {args.selected_benchmarks_json}"); exit(1)
    
    task_group_for_this_run = args.task_group 
    specific_benchmarks = selected_benchmarks_map.get(task_group_for_this_run)

    if specific_benchmarks is None: print(f"P{args.process_id}: No specific BMs for TG '{task_group_for_this_run}'. Map: {selected_benchmarks_map}. Skip.")
    elif not isinstance(specific_benchmarks, list) or not specific_benchmarks: print(f"P{args.process_id}: Invalid/empty BMs list for '{task_group_for_this_run}'. List: {specific_benchmarks}. Skip.")
    else:
        print(f"P{args.process_id}: Eval '{args.model_name}' on TG:'{task_group_for_this_run}', BMs:{specific_benchmarks} on GPU {args.gpu_id if torch.cuda.is_available() else 'CPU'}.")
        run_evaluation_for_model_and_tasks(
            input_model_name=args.model_name, task_group_to_evaluate=task_group_for_this_run,         
            selected_benchmarks_for_group=specific_benchmarks, process_id=args.process_id,
            physical_gpu_id=args.gpu_id, num_total_workers=args.num_gpus, 
            orchestrator_batch_size=args.batch_size)
    print(f"Proc {args.process_id}: Finished assigned work for this main.py instance.")