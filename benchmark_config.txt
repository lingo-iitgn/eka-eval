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
