BENCHMARK_CONFIG = {
   "CODE GENERATION": {
        "HumanEval": {
            "description": "Pass@k accuracy for HumanEval (Chen et al., 2021)",
            "evaluation_function": "code.humaneval.evaluate_humaneval",  
            "task_args": {
                "dataset_name": "openai_humaneval",
                "dataset_split": "test",
                "num_samples_per_task": 1,
                "use_fewshot": False,
                "max_new_tokens_completion": 384
            }
        },
        "MBPP": {
            "description": "Pass@k accuracy for MBPP (Austin et al., 2021)",
            "evaluation_function": "code.mbpp.evaluate_mbpp",
            "task_args": {
                "dataset_name_primary": "google-research-datasets/mbpp",
                "dataset_config_primary": "full",
                "dataset_name_fallback": "mbpp",
                "dataset_config_fallback": "sanitized",
                "dataset_split": "test",
                "num_samples_per_task": 1,
                "include_tests_in_prompt": False,
                "max_new_tokens_completion": 512
            }
        },
        "HumanEval+": {
            "description": "Pass@k accuracy for HumanEval+ (Liu et al., 2024a)",
            "evaluation_function": "code.humanevalplus.evaluate_humanevalplus",
            "task_args": {
                "dataset_name": "evalplus/humanevalplus",
                "dataset_split": "test",
                "num_samples_per_task": 1,
                "max_new_tokens_completion": 384
            }
        },
        "MBPP EvalPlus": {
            "description": "Pass@k accuracy for MBPP EvalPlus (Liu et al., 2024a)",
            "evaluation_function": "code.mbppplus.evaluate_mbpp_plus",
            "task_args": {
                "dataset_name_primary": "evalplus/mbppplus",
                "dataset_config_primary": None,
                "dataset_split": "test",
                "num_samples_per_task": 1,
                "max_new_tokens_completion": 512
            }
        }
   },
        
        "Tool use": {
            "API-Bank": {
            "description": "API function calling accuracy on API-Bank dataset",
            "evaluation_function": "tool_use.apibank.evaluate_apibank",
            "task_args": {
                "dataset_name": "liminghao1630/API-Bank",
                "dataset_split": "test",
                "max_new_tokens": 512,
                "generation_batch_size": 1,
                "num_few_shot": 0,
                "save_outputs": True,
                "resume": False
            }
        },
        "APIBench": {
            "description": "API call generation accuracy on APIBench dataset",
            "evaluation_function": "tool_use.apibench.evaluate_apibench",
            "task_args": {
                "dataset_name": "gorilla-llm/APIBench",
                "dataset_split": "test",
                "max_new_tokens": 256,
                "generation_batch_size": 1,
                "num_few_shot": 0,
                "save_outputs": True,
                "resume": False
            }
        }
        },
        
    "MATH": {
        "GSM8K": {
            "description": "Accuracy on GSM8K (Cobbe et al., 2021)",
            "evaluation_function": "math.gsm8k.evaluate_gsm8k",
            "task_args": {
                "dataset_name": "openai/gsm8k",
                "dataset_config_name": "main",
                "dataset_split": "test",
                "num_few_shot": 5,
                "max_new_tokens": 384,
                "generation_batch_size": 8
            }
        },
        "MATH": {
            "description": "Accuracy on Hendrycks MATH (Hendrycks et al., 2021b)",
            "evaluation_function": "math.math_eval.evaluate_math",
            "task_args": {
                "dataset_name": "hendrycks/competition_math",
                "dataset_split": "test",
                "num_few_shot": 4,
                "max_new_tokens": 512,
                "generation_batch_size": 4
            }
        },
        "GPQA": {
            "description": "Accuracy on GPQA (Rein et al., 2023)",
            "evaluation_function": "math.gpqa.evaluate_gpqa",
            "task_args": {
                "dataset_name": "Idavidrein/gpqa",
                "dataset_config": "gpqa_main",
                "dataset_split": "test",
                "num_few_shot": 3,
                "max_new_tokens": 10,
                "generation_batch_size": 8
            }
        },
        "ARC-Challenge": {
            "description": "Accuracy on ARC-Challenge (Clark et al., 2018)",
            "evaluation_function": "reasoning.arc_challenge.evaluate_arc_challenge",
            "task_args": {
                "dataset_name": "ai2_arc",
                "dataset_config_name": "ARC-Challenge",
                "dataset_split": "test",
                "max_new_tokens": 5,
                "generation_batch_size": 8
            }
        }
    },
    "READING COMPREHENSION": {
        "SQuAD": {
            "description": "F1 / Exact Match on SQuAD (Rajpurkar et al., 2018)",
            "evaluation_function": "reading_comprehension.squad.evaluate_squad",
            "task_args": {
                "dataset_name": "squad",
                "dataset_split": "validation",
                "max_new_tokens": 64,
                "generation_batch_size": 8,
                "checkpoint_dir": "checkpoints/squad_checkpoints",
                "resume": True,
                "checkpoint_save_interval_batches": 50
            }
        },
        "QuAC": {
            "description": "F1 / Exact Match on QuAC (Choi et al., 2018)",
            "evaluation_function": "reading_comprehension.quac.evaluate_quac",
            "task_args": {
                "dataset_name": "allenai/quac",
                "dataset_split": "validation",
                "max_new_tokens": 64,
                "generation_batch_size": 8,
                "checkpoint_dir": "checkpoints/quac_checkpoints",
                "resume": True,
                "checkpoint_save_interval_batches": 50
            }
        },
        "BoolQ": {
            "description": "Accuracy on BoolQ (Clark et al., 2019)",
            "evaluation_function": "reading_comprehension.boolq.evaluate_boolq",
            "task_args": {
                "dataset_name": "google/boolq",
                "dataset_split": "validation",
                "max_new_tokens": 10,
                "generation_batch_size": 16,
                "checkpoint_dir": "checkpoints/boolq_checkpoints",
                "resume": True,
                "checkpoint_save_interval_batches": 100
            }
        }
    },
    "COMMONSENSE REASONING": {
        "PIQA": {
        "description": "Accuracy on PIQA (Bisk et al., 2020)",
        "evaluation_function": "commonsense_reasoning.piqa.evaluate_piqa",
        "task_args": {
            "dataset_name": "piqa",
            "dataset_split": "validation",
            "max_new_tokens": 10,
            "generation_batch_size": 16,
            "num_few_shot": 5,
            "evaluation_method": "likelihood",  # Options: "likelihood" or "generation"
            "save_outputs": False,
            "disable_progress_bar": False,
            # Prompt template configuration
            "prompt_template_key_likelihood": "piqa_likelihood",
            "prompt_template_key_generation": "piqa_generation", 
            "prompt_template_key_few_shot_likelihood": "piqa_5shot_likelihood",
            "prompt_template_key_few_shot_generation": "piqa_5shot_generation",
            "prompt_file_benchmark_key": "piqa",
            "prompt_file_category": "reasoning"
        }
        },
        "SIQA": {
            "description": "Accuracy on SIQA (Sap et al., 2019)",
            "evaluation_function": "reasoning.siqa.evaluate_siqa",
            "task_args": {
                "dataset_name": "allenai/social_i_qa",
                "dataset_split": "validation",
                "max_new_tokens": 5,
                "generation_batch_size": 16
            }
        },
        "HellaSwag": {
            "description": "Accuracy on HellaSwag (Zellers et al., 2019a)",
            "evaluation_function": "reasoning.hellaswag.evaluate_hellaswag",
            "task_args": {
                "dataset_name": "hellaswag",
                "dataset_split": "validation",
                "max_new_tokens": 50,
                "generation_batch_size": 8
            }
        },
        "ARC-Easy": {
            "description": "Accuracy on ARC-Easy (Clark et al., 2018)",
            "evaluation_function": "reasoning.arc_easy.evaluate_arc_easy",
            "task_args": {
                "dataset_name": "ai2_arc",
                "dataset_config_name": "ARC-Easy",
                "dataset_split": "test",
                "max_new_tokens": 5,
                "generation_batch_size": 8
            }
        },
        "WinoGrande": {
            "description": "Accuracy on WinoGrande (Sakaguchi et al., 2021)",
            "evaluation_function": "reasoning.winogrande.evaluate_winogrande",
            "task_args": {
                "dataset_name": "winogrande",
                "dataset_config_name": "winogrande_xl",
                "dataset_split": "validation",
                "max_new_tokens": 5,
                "generation_batch_size": 16
            }
        },
        "CommonSenseQA": {
            "description": "Accuracy on CommonSenseQA (Talmor et al., 2018)",
            "evaluation_function": "reasoning.commonsenseqa.evaluate_commonsenseqa",
            "task_args": {
                "dataset_name": "commonsense_qa",
                "dataset_split": "validation",
                "num_few_shot": 7,
                "max_new_tokens": 5,
                "generation_batch_size": 8
            }
        },
        "OpenBookQA": {
            "description": "Accuracy on OpenBookQA (Mihaylov et al., 2018)",
            "evaluation_function": "reasoning.openbookqa.evaluate_openbookqa",
            "task_args": {
                "dataset_name": "allenai/openbookqa",
                "dataset_config_name": "main",
                "dataset_split": "test",
                "max_new_tokens": 5,
                "generation_batch_size": 8
            }
        }
    },
    "WORLD KNOWLEDGE": {
        "TriviaQA": {
            "description": "Accuracy on TriviaQA (Joshi et al., 2017)",
            "evaluation_function": "knowledge.triviaqa.evaluate_triviaqa",
            "task_args": {
                "dataset_name": "trivia_qa",
                "dataset_config_name": "rc.nocontext",
                "dataset_split": "validation",
                "num_few_shot": 5,
                "max_new_tokens": 50,
                "generation_batch_size": 8
            }
        },
        "NaturalQuestions": {
            "description": "Accuracy on NaturalQuestions (Kwiatkowski et al., 2019)",
            "evaluation_function": "knowledge.nq.evaluate_nq",
            "task_args": {
                "dataset_name": "nq_open",
                "dataset_split": "validation",
                "num_few_shot": 5,
                "max_new_tokens": 50,
                "generation_batch_size": 8
            }
        }
    },
    "LONG CONTEXT": {
        "ZeroSCROLLS": {
            "description": "ROUGE, F1, Acc on ZeroSCROLLS (Shaham et al., 2023)",
            "evaluation_function": "long_context.zeroscrolls.evaluate_zeroscrolls",
            "task_args": {}
        },
        "NeedleInAHaystack": {
            "description": "Retrieval Acc on Needle-in-a-Haystack (Kamradt, 2023)",
            "evaluation_function": "long_context.niah.evaluate_niah",
            "task_args": {}
        },
        "InfiniteBench": {
            "description": "Task-Acc on InfiniteBench (Zhang et al., 2024)",
            "evaluation_function": "long_context.infinitebench.evaluate_infinitebench",
            "task_args": {}
        }
    },
    "General":
        {
    "MMLU": {
        "description": "5-shot Accuracy on MMLU (Hendrycks et al., 2021a)",
        "evaluation_function": "general.mmlu.evaluate_mmlu",
        "task_args": {
            "dataset_name": "cais/mmlu",
            "dataset_config_name": "all",
            "dataset_split": "test",
            "num_few_shot": 5,
            "max_new_tokens": 5,
            "generation_batch_size": 8
        }
    },
    "MMLU-Pro": {
        "description": "5-shot Accuracy on MMLU-Pro (Wang et al., 2024b)",
        "evaluation_function": "general.mmlu_pro.evaluate_mmlu_pro",
        "task_args": {
            "dataset_name": "TIGER-Lab/MMLU-Pro",
            "dataset_split": "test",
            "num_few_shot": 5,
            "max_new_tokens": 5,
            "generation_batch_size": 8
        }
    },
    "IFEval": {
        "description": "Instruction Following Accuracy on IFEval (Zhou et al., 2023)",
        "evaluation_function": "general.ifeval.evaluate_ifeval",
        "task_args": {
            "dataset_name": "google/ifeval",
            "dataset_split": "test"
        }
    },
    "BBH": {
        "description": "3-shot Accuracy on a BBH subtask (Suzgun et al., 2022)",
        "evaluation_function": "general.bbh.evaluate_bbh",
        "task_args": {
            "dataset_name": "lukaemon/bbh",
            "dataset_config_name": "boolean_expressions",
            "dataset_split": "test",
            "num_few_shot": 3,
            "max_new_tokens": 10,
            "generation_batch_size": 8
        }
    },
    "AGIEval": {
        "description": "3–5 shot Accuracy on an AGIEval subtask (Zhong et al., 2023)",
        "evaluation_function": "general.agieval.evaluate_agieval",
        "task_args": {
            "dataset_name": "hails/agieval",
            "dataset_config_name": "aqua-rat",
            "dataset_split": "test",
            "num_few_shot": 3,
            "max_new_tokens": 10,
            "generation_batch_size": 8
        }
    } },
    "INDIC BENCHMARKS": {
        "MMLU-IN": {
            "description": "Accuracy on MMLU-Indic",
            "evaluation_function": "indic.mmlu_in.evaluate_mmlu_in",
            "task_args": {
                "dataset_name": "sarvamai/mmlu-indic",
                "dataset_split": "validation",
                "num_few_shot": 0,
                "max_new_tokens": 5
            }
        },
        "TriviaQA-IN": {
            "description": "Accuracy",
            "evaluation_function": "indic.triviaqa_in.evaluate_triviaqa_in",
            "task_args": {}
        },
        "MILU": {
            "description": "Accuracy",
            "evaluation_function": "indic.milu.evaluate_milu",
            "task_args": {}
        },
        "GSM-8K-IN": {
            "description": "Accuracy",
            "evaluation_function": "indic.gsm8k_in.evaluate_gsm8k_in",
            "task_args": {}
        },
        "CROSS SUM": {
            "description": "Accuracy",
            "evaluation_function": "indic.cross_sum.evaluate_cross_sum",
            "task_args": {}
        },
        "BOOLQ-IN": {
            "description": "Accuracy on Indic BoolQ dataset",
            "evaluation_function": "indic.boolq_in.evaluate_boolq_in",
            "task_args": {
                "dataset_name": "sarvamai/boolq-indic",
                "target_languages": ["en", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
                "dataset_split": "validation",
                "max_new_tokens": 10
            }
        },
            "ARC-Challenge-Indic": {
        "description": "Zero-shot ARC-Challenge-Indic evaluation across 11 languages",
        "evaluation_function": "indic.arc_c_in.evaluate_arc_c_in",
        "task_args": {
            "dataset_name": "sarvamai/arc-challenge-indic",
            "target_languages": ["bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
            "dataset_split": "validation[:50]",
            "num_few_shot": 0,
            "max_new_tokens": 10,
            "generation_batch_size": 8,
            "prompt_template_name_zeroshot": "arc_c_in_0shot",
            "prompt_template_name_fewshot": "arc_c_in_5shot",
            "prompt_file_benchmark_key": "arc_c_in",
            "prompt_file_category": "indic",
        }
    },
        "Flores-IN": {
            "description": "BLEU, ChrF",
            "evaluation_function": "indic.flores_in.evaluate_flores_in",
            "task_args": {}
        },
        "XQuAD-IN": {
            "description": "F1 / Exact Match",
            "evaluation_function": "indic.xquad_in.evaluate_xquad_in",
            "task_args": {}
        },
        "XorQA-IN": {
            "description": "F1 / Exact Match",
            "evaluation_function": "indic.xorqa_in.evaluate_xorqa_in",
            "task_args": {}
        }
    }
}
