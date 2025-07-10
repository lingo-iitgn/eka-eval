import pandas as pd
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json 
from datetime import datetime
import gc
import logging
from typing import List, Dict
import argparse
from eval_tasks.code.humaneval import evaluate_humaneval 
from eval_tasks.code.mbpp import evaluate_mbpp
from eval_tasks.math.gsm8k import evaluate_gsm8k
from eval_tasks.math.math import evaluate_math
from eval_tasks.reading_comprehension.boolq import evaluate_boolq 
from eval_tasks.reading_comprehension.squad import evaluate_squad
from eval_tasks.reading_comprehension.quac import evaluate_quac
BENCHMARK_CONFIG = {
    "CODE GENERATION": {
        "HumanEval": {"description": "Pass@1 accuracy (Chen et al., 2021)", "evaluation_function": "evaluate_humaneval"},
        "MBPP": {"description": "Pass@1 accuracy (Austin et al., 2021)", "evaluation_function": "evaluate_mbpp"},
        "HumanEval+": {"description": "Pass@1 accuracy (Liu et al., 2024a)", "evaluation_function": "evaluate_humaneval_plus"},
        "MBPP EvalPlus": {"description": "Pass@1 accuracy (Liu et al., 2024a)", "evaluation_function": "evaluate_mbpp_evalplus"},
        "MultiPL-E": {"description": "Pass@1 accuracy (Cassano et al., 2023)", "evaluation_function": "evaluate_multipl_e"}
    },
    "MATH AND REASONING": {
        "GSM8K": {"description": "Accuracy (Cobbe et al., 2021)", "evaluation_function": "evaluate_gsm8k"},
        "MATH": {"description": "Accuracy (Hendrycks et al., 2021b)", "evaluation_function": "evaluate_math"},
        "GPQA": {"description": "Accuracy (Rein et al., 2023)", "evaluation_function": "evaluate_gpqa"},
        "ARC-Challenge": {"description": "Accuracy (Clark et al., 2018)", "evaluation_function": "evaluate_arc_challenge"}
    },
    "READING COMPREHENSION": {
        "SQuAD": {"description": "F1 / Exact Match (Rajpurkar et al., 2018)", "evaluation_function": "evaluate_squad"},
        "QuAC": {"description": "F1 / Exact Match (Choi et al., 2018)", "evaluation_function": "evaluate_quac"},
        "BoolQ": {"description": "Accuracy (Clark et al., 2019)", "evaluation_function": "evaluate_boolq"}
    },
    "COMMONSENSE REASONING": {
        "PIQA": {"description": "Accuracy (Bisk et al., 2020)", "evaluation_function": "evaluate_piqa"},
        "SIQA": {"description": "Accuracy (Sap et al., 2019)", "evaluation_function": "evaluate_siqa"},
        "HellaSwag": {"description": "Accuracy (Zellers et al., 2019a)", "evaluation_function": "evaluate_hellaswag"},
        "ARC-Easy": {"description": "Accuracy (Clark et al., 2018)", "evaluation_function": "evaluate_arc_easy"},
        "ARC-Challenge": {"description": "Accuracy (Clark et al., 2018)", "evaluation_function": "evaluate_arc_challenge"},
        "WinoGrande": {"description": "Accuracy (Sakaguchi et al., 2021)", "evaluation_function": "evaluate_winogrande"},
        "CommonSenseQA": {"description": "7-shot Accuracy (Talmor et al., 2018)", "evaluation_function": "evaluate_commonsenseqa"},
        "OpenBookQA": {"description": "Accuracy (Mihaylov et al., 2018)", "evaluation_function": "evaluate_openbookqa"}
    },
    "WORLD KNOWLEDGE": {
        "TriviaQA": {"description": "5-shot Accuracy (Joshi et al., 2017)", "evaluation_function": "evaluate_triviaqa"},
        "NaturalQuestions": {"description": "5-shot Accuracy (Kwiatkowski et al., 2019)", "evaluation_function": "evaluate_nq"}
    },
    "TOOL-USE": {
        "Nexus": {"description": "Success Rate / Task-specific metrics (Srinivasan et al., 2023)", "evaluation_function": "evaluate_nexus"},
        "API-Bank": {"description": "API Call Accuracy, ROUGE-L (Li et al., 2023b)", "evaluation_function": "evaluate_apibank"},
        "API-Bench": {"description": "API Recommendation Accuracy (Patil et al., 2023)", "evaluation_function": "evaluate_apibench"},
        "BFCL": {"description": "Function Call Accuracy, Latency (Yan et al., 2024)", "evaluation_function": "evaluate_bfcl"}
    },
    "LONG CONTEXT": {
        "ZeroSCROLLS": {"description": "ROUGE, F1, Accuracy, Exponential Similarity (Shaham et al., 2023)", "evaluation_function": "evaluate_zeroscrolls"},
        "Needle-in-a-Haystack": {"description": "Retrieval Accuracy, Recall (Kamradt, 2023)", "evaluation_function": "evaluate_needle"},
        "InfiniteBench": {"description": "Task-specific Accuracy, Recall (Zhang et al., 2024)", "evaluation_function": "evaluate_infinitebench"}
    },
    "GENERAL": {
        "MMLU": {"description": "5-shot Accuracy (Hendrycks et al., 2021a)", "evaluation_function": "evaluate_mmlu"},
        "MMLU-Pro": {"description": "5-shot Accuracy (Wang et al., 2024b)", "evaluation_function": "evaluate_mmlu_pro"},
        "IFEval": {"description": "Accuracy (Zhou et al., 2023)", "evaluation_function": "evaluate_ifeval"},
        "BBH": {"description": "3-shot Accuracy (Suzgun et al., 2022)", "evaluation_function": "evaluate_bbh"},
        "AGIEval": {"description": "3–5 shot Accuracy (Zhong et al., 2023)", "evaluation_function": "evaluate_agieval"}
    },
    "INDIC BENCHMARKS": {
        "MMLU-IN": {"description": "Accuracy", "evaluation_function": "evaluate_mmlu_in"},
        "TriviaQA-IN": {"description": "Accuracy", "evaluation_function": "evaluate_triviaqa_in"},
        "MILU": {"description": "Accuracy", "evaluation_function": "evaluate_milu"},
        "GSM-8K-IN": {"description": "Accuracy", "evaluation_function": "evaluate_gsm8k_in"},
        "CROSS SUM": {"description": "Accuracy", "evaluation_function": "evaluate_crosssum"},
        "BOOLQ": {"description": "Accuracy", "evaluation_function": "evaluate_boolq_in"},
        "ARC-IN": {"description": "Accuracy", "evaluation_function": "evaluate_arc_in"},
        "Flores-IN": {"description": "BLEU, ChrF", "evaluation_function": "evaluate_flores_in"},
        "XQuAD-IN": {"description": "F1 / Exact Match", "evaluation_function": "evaluate_xquad_in"},
        "XorQA-IN": {"description": "F1 / Exact Match", "evaluation_function": "evaluate_xorqa_in"}
    }
}

PRE_CALCULATED_RESULTS = pd.DataFrame() 


# placeholders for now
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

# initialise function for the model
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
    selected_benchmarks_for_group: List[str], # from run_benchmarks.py
    process_id: int,
    physical_gpu_id: int,
    num_total_workers: int, 
    orchestrator_batch_size: int ##
):
    global PRE_CALCULATED_RESULTS # taken from above pre calculated results
    model_name_lower = input_model_name.lower()
    csv_file_path = 'calculated.csv'
    if os.path.exists(csv_file_path):
        try:
            temp_df = pd.read_csv(csv_file_path)
            if 'Model' in temp_df.columns: 
                PRE_CALCULATED_RESULTS = temp_df[temp_df['Model'].str.lower() == model_name_lower].copy()
            else: 
                PRE_CALCULATED_RESULTS = pd.DataFrame()
        except Exception as e: 
            print(f"P{process_id} Warn: Load pre-calc fail: {e}"); PRE_CALCULATED_RESULTS = pd.DataFrame()
    else: 
        PRE_CALCULATED_RESULTS = pd.DataFrame()

    model_pipeline, model_param_count_str = initialize_pipeline(input_model_name, device_id=0) #calling and initialising on gpu
    if model_pipeline is None: 
        print(f"P{process_id}: Failedd {input_model_name}. Skip."); return # have to change print statements to logs
    
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
                args_eval.update({"batch_size":8, "dataset_split":"validation[:10]", "process_id":process_id, "gpu_id":physical_gpu_id, "num_gpus":num_total_workers, "checkpoint_dir":cp_dir_squad, "resume":True})
            elif eval_fn_name == "evaluate_quac": # Added QuAC
                cp_dir_quac = os.path.join("checkpoints", "quac_checkpoints"); os.makedirs(cp_dir_quac, exist_ok=True)
                args_eval.update({"batch_size":8, "dataset_split":"validation[:10]", "process_id":process_id, "gpu_id":physical_gpu_id, "num_gpus":num_total_workers, "checkpoint_dir":cp_dir_quac, "resume":True})
            elif eval_fn_name == "evaluate_gsm8k":
                args_eval.update({"batch_size":8,"dataset_split":"test[:10]","process_id":process_id,"gpu_id":physical_gpu_id,"num_gpus":num_total_workers,"resume":True}) #resyme?1
            elif eval_fn_name == "evaluate_math": 
                #math_cp_dir = os.path.join("checkpoints", "hendrycks_math_checkpoints"); os.makedirs(math_cp_dir, exist_ok=True)
                #args_eval["checkpoint_dir"] = math_cp_dir
                args_eval.update({"batch_size":8,"dataset_split":"test[:10]","process_id":process_id,"gpu_id":physical_gpu_id,"resume":True})
            
            scores_dict_from_eval = {} 
            try:
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
            if pd.notna(current_score): 
                scores_for_avg.append(float(current_score))
            else: 
                print(f"Warn: No valid score for {task_group_name}-{bm_name} from '{eval_fn_name}'. Dict: {scores_dict_from_eval}")
        except Exception as e: 
            print(f"\033[91mP{process_id}: Outer error during evaluation of {task_group_name}-{bm_name}: {e}\033[0m"); 
            import traceback; traceback.print_exc() #?!
            new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':bm_name,'Score':pd.NA,'Timestamp':datetime.now().isoformat()})
    
    if not is_single_bm_grp and len(scores_for_avg) > 1:
        avg = sum(scores_for_avg)/len(scores_for_avg); print(f"P{process_id}: Avg for {task_group_name} ({len(scores_for_avg)} scores): {avg:.2f}")
        new_results.append({'Model':input_model_name,'Size (B)':model_param_count_str,'Task':task_group_name,'Benchmark':'Average','Score':avg,'Timestamp':datetime.now().isoformat()})
    elif not is_single_bm_grp and scores_for_avg: 
        print(f"P{process_id}: Only one score ({scores_for_avg[0]:.2f}) for {task_group_name}. No 'Average' stored.")

    # Save all new results when calc live
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
    parser = argparse.ArgumentParser(description="LLM Benchmarks.") # arguments are there so to override 
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