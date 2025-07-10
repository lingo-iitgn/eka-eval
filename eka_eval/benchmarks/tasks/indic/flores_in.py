#!/usr/bin/env python3
"""
Complete High-Accuracy FLORES Translation Evaluation Script
Enhanced with comprehensive prompt engineering and configuration-driven evaluation
"""

import os
import sys
import json
import re
import traceback
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu import corpus_chrf
from accelerate import Accelerator, DistributedDataParallelKwargs

# Configuration Constants
MODEL_NAME = "sarvamai/sarvam-1"
DATASET_NAME = "google/IndicGenBench_flores_in"
OUTPUT_DIR = "flores_enhanced_eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_INDIC_LANG_CODES = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
TRANSLATION_DIRECTION = "enxx"  # enxx for en->indic, xxen for indic->en

# Enhanced Configuration
@dataclass
class FloresConfig:
    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    translation_direction: str = TRANSLATION_DIRECTION
    target_languages: List[str] = None
    num_samples_per_lang: int = 100
    batch_size: int = 4
    max_new_tokens: int = 128
    output_dir: str = OUTPUT_DIR
    use_few_shot: bool = True
    num_few_shot_examples: int = 3
    
    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = TARGET_INDIC_LANG_CODES

class FloresPromptManager:
    """Enhanced prompt management with comprehensive few-shot examples"""
    
    def __init__(self, config: FloresConfig):
        self.config = config
        self.language_mappings = {
            "as": "Assamese", "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
            "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "or": "Odia", 
            "pa": "Punjabi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu", "en": "English"
        }
        self.load_prompt_config()
    
    def load_prompt_config(self):
        """Load comprehensive prompt configuration with all language pairs"""
        # Comprehensive few-shot examples for each language pair
        self.few_shot_examples = {
            "en_to_hi": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Hindi",
                    "target_text": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।"
                },
                {
                    "source_lang": "English", 
                    "source_text": "I love learning new languages.",
                    "target_lang": "Hindi",
                    "target_text": "मुझे नई भाषाएं सीखना पसंद है।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Hindi", 
                    "target_text": "तकनीक तेज़ी से आगे बढ़ रही है।"
                }
            ],
            "en_to_bn": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Bengali",
                    "target_text": "দ্রুত বাদামী শিয়াল অলস কুকুরের ওপর দিয়ে লাফ দেয়।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Bengali", 
                    "target_text": "আমি নতুন ভাষা শিখতে ভালোবাসি।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Bengali",
                    "target_text": "প্রযুক্তি দ্রুত এগিয়ে চলেছে।"
                }
            ],
            "en_to_gu": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Gujarati",
                    "target_text": "ઝડપી ભૂરો શિયાળ આળસુ કુતરાની ઉપર કૂદે છે।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Gujarati",
                    "target_text": "મને નવી ભાષાઓ શીખવાનું ગમે છે।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Gujarati",
                    "target_text": "ટેકનોલોજી ઝડપથી આગળ વધી રહી છે।"
                }
            ],
            "en_to_ta": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Tamil",
                    "target_text": "வேகமான பழுப்பு நிற நரி சோம்பேறி நாயின் மேல் குதிக்கிறது।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Tamil",
                    "target_text": "புதிய மொழிகளைக் கற்றுக்கொள்வது எனக்குப் பிடிக்கும்।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Tamil",
                    "target_text": "தொழில்நுட்பம் வேகமாக முன்னேறி வருகிறது।"
                }
            ],
            "en_to_te": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Telugu",
                    "target_text": "వేగంగా గోధుమ రంగు నక్క సోమరి కుక్క మీదుగా దూకుతుంది।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Telugu",
                    "target_text": "నేను కొత్త భాషలు నేర్చుకోవడం ఇష్టపడతాను।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Telugu",
                    "target_text": "సాంకేతికత వేగంగా అభివృద్ధి చెందుతోంది।"
                }
            ],
            "en_to_kn": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Kannada",
                    "target_text": "ವೇಗವಾದ ಕಂದು ನರಿ ಆಲಸ್ಯದ ನಾಯಿಯ ಮೇಲೆ ಜಿಗಿಯುತ್ತದೆ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Kannada",
                    "target_text": "ನಾನು ಹೊಸ ಭಾಷೆಗಳನ್ನು ಕಲಿಯಲು ಇಷ್ಟಪಡುತ್ತೇನೆ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Kannada",
                    "target_text": "ತಂತ್ರಜ್ಞಾನವು ವೇಗವಾಗಿ ಮುಂದುವರಿಯುತ್ತಿದೆ।"
                }
            ],
            "en_to_ml": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Malayalam",
                    "target_text": "വേഗതയുള്ള തവിട്ടുനിറത്തിലുള്ള കുറുക്കൻ മടിയനായ നായയുടെ മുകളിലൂടെ ചാടുന്നു।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Malayalam",
                    "target_text": "പുതിയ ഭാഷകൾ പഠിക്കാൻ ഞാൻ ഇഷ്ടപ്പെടുന്നു।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Malayalam",
                    "target_text": "സാങ്കേതികവിദ്യ അതിവേഗത്തിൽ മുന്നേറുകയാണ്।"
                }
            ],
            "en_to_mr": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Marathi",
                    "target_text": "वेगवान तपकिरी कोल्हा आळशी कुत्र्यावर उडी मारतो।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Marathi",
                    "target_text": "मला नवीन भाषा शिकायला आवडतात।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Marathi",
                    "target_text": "तंत्रज्ञान वेगाने प्रगती करत आहे।"
                }
            ],
            "en_to_pa": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Punjabi",
                    "target_text": "ਤੇਜ਼ ਭੂਰੀ ਲੂੰਬੜੀ ਆਲਸੀ ਕੁੱਤੇ ਦੇ ਉੱਪਰੋਂ ਕੁੱਦਦੀ ਹੈ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Punjabi",
                    "target_text": "ਮੈਨੂੰ ਨਵੀਆਂ ਭਾਸ਼ਾਵਾਂ ਸਿੱਖਣਾ ਪਸੰਦ ਹੈ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Punjabi",
                    "target_text": "ਤਕਨਾਲੋਜੀ ਤੇਜ਼ੀ ਨਾਲ ਅੱਗੇ ਵਧ ਰਹੀ ਹੈ।"
                }
            ],
            "en_to_or": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Odia",
                    "target_text": "ଦ୍ରୁତ ବାଦାମୀ ଶିଆଳ ଅଳସୁଆ କୁକୁର ଉପରେ ଡେଇଁପଡ଼େ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Odia",
                    "target_text": "ମୁଁ ନୂତନ ଭାଷା ଶିଖିବାକୁ ଭଲପାଏ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Odia",
                    "target_text": "ଟେକ୍ନୋଲୋଜି ଦ୍ରୁତ ଗତିରେ ଆଗେଇ ଚାଲିଛି।"
                }
            ],
            "en_to_as": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Assamese",
                    "target_text": "দ্ৰুত মুগা শিয়ালটোৱে আলসুৱা কুকুৰটোৰ ওপৰেদি জঁপিয়াই যায়।"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Assamese",
                    "target_text": "মই নতুন ভাষা শিকি ভাল পাওঁ।"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Assamese",
                    "target_text": "প্ৰযুক্তি দ্ৰুতগতিত আগবাঢ়িছে।"
                }
            ],
            "en_to_ur": [
                {
                    "source_lang": "English",
                    "source_text": "The quick brown fox jumps over the lazy dog.",
                    "target_lang": "Urdu",
                    "target_text": "تیز بھورا لومڑی سست کتے کے اوپر سے کودتا ہے۔"
                },
                {
                    "source_lang": "English",
                    "source_text": "I love learning new languages.",
                    "target_lang": "Urdu",
                    "target_text": "مجھے نئی زبانیں سیکھنا پسند ہے۔"
                },
                {
                    "source_lang": "English",
                    "source_text": "Technology is advancing rapidly.",
                    "target_lang": "Urdu",
                    "target_text": "ٹیکنالوجی تیزی سے ترقی کر رہی ہے۔"
                }
            ]
        }
        
        # Generation configuration optimized for translation
        self.generation_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "num_beams": 4,
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "early_stopping": True,
            "no_repeat_ngram_size": 3
        }
        
        # Post-processing cleanup patterns
        self.cleanup_patterns = [
            r'^(English|Hindi|Bengali|Gujarati|Tamil|Telugu|Kannada|Malayalam|Marathi|Punjabi|Odia|Assamese|Urdu):\s*',
            r'^(Translation|Output|Result|Answer):\s*',
            r'\n\n.*$',
            r'\n[A-Z][a-z]+:.*$'
        ]
    
    def get_few_shot_examples(self, source_lang: str, target_lang: str) -> List[Dict]:
        """Get few-shot examples for language pair"""
        if not self.config.use_few_shot:
            return []
        
        examples_key = f"{source_lang}_to_{target_lang}"
        examples = self.few_shot_examples.get(examples_key, [])
        return examples[:self.config.num_few_shot_examples]
    
    def create_translation_prompt(self, source_text: str, source_lang: str, target_lang: str) -> str:
        """Create enhanced translation prompt with few-shot examples"""
        source_lang_full = self.language_mappings.get(source_lang, source_lang.capitalize())
        target_lang_full = self.language_mappings.get(target_lang, target_lang.capitalize())
        
        # Get few-shot examples
        few_shot_examples = self.get_few_shot_examples(source_lang, target_lang)
        
        # Build prompt with examples
        prompt_parts = []
        
        # Add few-shot examples
        for example in few_shot_examples:
            example_prompt = f"{example['source_lang']}: {example['source_text']}\n{example['target_lang']}: {example['target_text']}"
            prompt_parts.append(example_prompt)
        
        # Add main query
        main_prompt = f"{source_lang_full}: {source_text}\n{target_lang_full}:"
        prompt_parts.append(main_prompt)
        
        return "\n\n".join(prompt_parts)

class FloresDataManager:
    """Enhanced data loading and management"""
    
    def __init__(self, config: FloresConfig):
        self.config = config
    
    def load_flores_data_robust(self, indic_lang_code: str, translation_direction: str) -> HFDataset:
        """Robust data loading with multiple fallback strategies"""
        cache_file = f"flores_{indic_lang_code}_{translation_direction}_test.json"
        
        # Try loading from cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                return HFDataset.from_list(examples)
            except Exception as e:
                print(f"Failed to load cache {cache_file}: {e}")
        
        # Try multiple loading strategies
        loading_strategies = [
            lambda: load_dataset(self.config.dataset_name, trust_remote_code=True),
            lambda: load_dataset(self.config.dataset_name, trust_remote_code=True, download_mode="force_redownload"),
            lambda: load_dataset(self.config.dataset_name, data_files={
                "loaded_data_split": f"flores_{indic_lang_code}_{translation_direction}_test.json"
            }, trust_remote_code=True)
        ]
        
        dataset = None
        for i, strategy in enumerate(loading_strategies):
            try:
                print(f"    Trying loading method {i+1}...")
                dataset = strategy()
                print(f"    ✅ Method {i+1} successful!")
                break
            except Exception as e:
                print(f"    ❌ Method {i+1} failed: {e}")
                continue
        
        if dataset is None:
            raise ValueError("All dataset loading methods failed")
        
        # Extract relevant examples
        extracted_examples = []
        
        # Handle different dataset structures
        splits_to_check = ["loaded_data_split"] + list(dataset.keys()) if isinstance(dataset, dict) else ["train"]
        
        for split_name in splits_to_check:
            if split_name in dataset:
                for item in dataset[split_name]:
                    if self._is_relevant_example(item, indic_lang_code, translation_direction):
                        extracted_examples.append(self._extract_example_data(item))
        
        if not extracted_examples:
            raise ValueError(f"No examples found for {indic_lang_code}-{translation_direction}")
        
        print(f"    Found {len(extracted_examples)} examples for {indic_lang_code}")
        
        # Cache for future use
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_examples, f, ensure_ascii=False, indent=2)
        
        return HFDataset.from_list(extracted_examples)
    
    def _is_relevant_example(self, item: Dict, lang_code: str, direction: str) -> bool:
        """Check if example is relevant for the language and direction"""
        if 'examples' in item and isinstance(item['examples'], dict):
            examples = item['examples']
            return (examples.get('lang') == lang_code and 
                   examples.get('translation_direction') == direction)
        return False
    
    def _extract_example_data(self, item: Dict) -> Dict:
        """Extract example data from dataset item"""
        if 'examples' in item:
            return item['examples']
        return item

class FloresTranslationEngine:
    """Enhanced translation engine with optimized generation"""
    
    def __init__(self, config: FloresConfig, accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.prompt_manager = FloresPromptManager(config)
    
    def batch_translate(self, model, tokenizer, prompts: List[str]) -> List[str]:
        """Perform optimized batch translation"""
        translations = []
        
        with tqdm(total=len(prompts), desc="Translating", disable=not self.accelerator.is_main_process) as pbar:
            for i in range(0, len(prompts), self.config.batch_size):
                batch_prompts = prompts[i:i + self.config.batch_size]
                batch_translations = []
                
                for prompt in batch_prompts:
                    try:
                        translation = self._translate_single(model, tokenizer, prompt)
                        batch_translations.append(translation)
                    except Exception as e:
                        print(f"Translation error: {e}")
                        batch_translations.append("[TRANSLATION ERROR]")
                
                translations.extend(batch_translations)
                pbar.update(len(batch_prompts))
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return translations
    
    def _translate_single(self, model, tokenizer, prompt: str) -> str:
        """Translate a single prompt with enhanced post-processing"""
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **self.prompt_manager.generation_config,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return self._clean_translation_enhanced(translation)
    
    def _clean_translation_enhanced(self, text: str) -> str:
        """Enhanced translation cleaning with configurable patterns"""
        if not text:
            return text
        
        # Apply cleanup patterns
        for pattern in self.prompt_manager.cleanup_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Take first line if multiple lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            text = lines[0]
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Apply length constraints
        max_len = 512
        min_len = 3
        
        if len(text) > max_len:
            text = text[:max_len].rsplit(' ', 1)[0]  # Cut at word boundary
        
        if len(text) < min_len:
            return "[INVALID_TRANSLATION]"
        
        return text.strip()

class FloresEvaluator:
    """Enhanced evaluation with comprehensive metrics and analysis"""
    
    def __init__(self, config: FloresConfig):
        self.config = config
        self.data_manager = FloresDataManager(config)
        self.accelerator = self._setup_accelerator()
        self.translation_engine = FloresTranslationEngine(config, self.accelerator)
        
    def _setup_accelerator(self) -> Accelerator:
        """Setup accelerator for multi-GPU evaluation"""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(
            mixed_precision="fp16",
            kwargs_handlers=[ddp_kwargs]
        )
        return accelerator
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load and prepare model and tokenizer"""
        if self.accelerator.is_main_process:
            print(f"Loading model: {self.config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={i: "20GB" for i in range(torch.cuda.device_count())}
        )
        
        model.eval()
        model, tokenizer = self.accelerator.prepare(model, tokenizer)
        
        if self.accelerator.is_main_process:
            print(f"Model loaded successfully on device: {next(model.parameters()).device}")
        
        return model, tokenizer
    
    def evaluate_language_pair(self, model, tokenizer, indic_lang_code: str) -> Tuple[Dict, List[Dict]]:
        """Evaluate translation for a specific language pair"""
        source_lang, target_lang = self._get_language_pair(indic_lang_code)
        pair_name = f"flores_{source_lang}-{target_lang}"
        
        if self.accelerator.is_main_process:
            print(f"\n--- Evaluating: {source_lang} to {target_lang} ---")
        
        try:
            # Load dataset
            dataset = self.data_manager.load_flores_data_robust(indic_lang_code, self.config.translation_direction)
            
            num_samples = min(self.config.num_samples_per_lang, len(dataset))
            if num_samples == 0:
                return {"chrf++": 0.0, "info": "No samples"}, []
            
            subset = dataset.select(range(num_samples))
            
            # Prepare data
            prompts, references, detailed_logs = self._prepare_evaluation_data(
                subset, source_lang, target_lang, pair_name
            )
            
            if not prompts:
                return {"chrf++": 0.0, "info": "No valid examples"}, []
            
            if self.accelerator.is_main_process:
                print(f"  Translating {len(prompts)} examples...")
            
            # Perform translation
            predictions = self.translation_engine.batch_translate(model, tokenizer, prompts)
            
            # Update detailed logs with predictions
            for i, prediction in enumerate(predictions):
                if i < len(detailed_logs):
                    detailed_logs[i]["model_translation"] = prediction
            
            # Calculate metrics
            result = self._calculate_metrics(predictions, references, detailed_logs)
            
            # Save results
            if self.accelerator.is_main_process:
                self._save_language_results(pair_name, detailed_logs, result)
            
            return result, detailed_logs
            
        except Exception as e:
            error_msg = f"Error processing {pair_name}: {str(e)}"
            if self.accelerator.is_main_process:
                print(f"  ERROR: {error_msg}")
                traceback.print_exc()
            return {"chrf++": 0.0, "error": error_msg}, []
    
    def _get_language_pair(self, indic_lang_code: str) -> Tuple[str, str]:
        """Get source and target language codes"""
        if self.config.translation_direction == "enxx":
            return "en", indic_lang_code
        elif self.config.translation_direction == "xxen":
            return indic_lang_code, "en"
        else:
            raise ValueError(f"Unsupported translation direction: {self.config.translation_direction}")
    
    def _prepare_evaluation_data(self, subset: HFDataset, source_lang: str, target_lang: str, pair_name: str) -> Tuple[List[str], List[str], List[Dict]]:
        """Prepare data for evaluation with enhanced logging"""
        prompts = []
        references = []
        detailed_logs = []
        
        for example_idx, example_data in tqdm(
            enumerate(subset),
            total=len(subset),
            desc=f"Preparing {pair_name} examples",
            disable=not self.accelerator.is_main_process
        ):
            source_text = example_data.get("source", "")
            reference_translation = example_data.get("target", "")
            
            # Validate data
            if not isinstance(source_text, str) or not isinstance(reference_translation, str):
                continue
            
            if not source_text.strip() or not reference_translation.strip():
                continue
            
            # Create prompt
            prompt = self.translation_engine.prompt_manager.create_translation_prompt(
                source_text, source_lang, target_lang
            )
            
            prompts.append(prompt)
            references.append(reference_translation)
            
            # Create detailed log entry
            detailed_logs.append({
                "pair_key": pair_name,
                "example_index": example_idx,
                "source_text": source_text,
                "reference_translation": reference_translation,
                "model_translation": "",  # Will be filled later
                "prompt_used": prompt,
                "source_lang": source_lang,
                "target_lang": target_lang
            })
        
        return prompts, references, detailed_logs
    
    def _calculate_metrics(self, predictions: List[str], references: List[str], detailed_logs: List[Dict]) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        # Filter valid predictions
        valid_pairs = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if pred.strip() and ref.strip() and pred != "[TRANSLATION ERROR]" and pred != "[INVALID_TRANSLATION]":
                valid_pairs.append((pred, ref))
        
        if not valid_pairs:
            return {
                "chrf++": 0.0,
                "valid_samples": 0,
                "total_samples": len(predictions),
                "info": "No valid predictions after filtering"
            }
        
        valid_predictions, valid_references = zip(*valid_pairs)
        
        # Calculate chrF++ score
        references_list = [[ref] for ref in valid_references]
        
        try:
            chrf_score = corpus_chrf(
                valid_predictions,
                references_list,
                word_order=2,
                beta=2
            ).score
        except Exception as e:
            print(f"Error calculating chrF++ score: {e}")
            chrf_score = 0.0
        
        # Calculate additional metrics
        result = {
            "chrf++": chrf_score,
            "valid_samples": len(valid_pairs),
            "total_samples": len(predictions),
            "validity_rate": len(valid_pairs) / len(predictions) if predictions else 0.0
        }
        
        # Add quality analysis
        if detailed_logs:
            result["quality_analysis"] = self._analyze_translation_quality(valid_predictions, valid_references)
        
        return result
    
    def _analyze_translation_quality(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze translation quality metrics"""
        analysis = {
            "avg_prediction_length": np.mean([len(pred.split()) for pred in predictions]),
            "avg_reference_length": np.mean([len(ref.split()) for ref in references]),
            "length_ratio": 0.0,
            "empty_predictions": sum(1 for pred in predictions if not pred.strip()),
            "very_short_predictions": sum(1 for pred in predictions if len(pred.split()) < 3)
        }
        
        if analysis["avg_reference_length"] > 0:
            analysis["length_ratio"] = analysis["avg_prediction_length"] / analysis["avg_reference_length"]
        
        return analysis
    
    def _save_language_results(self, pair_name: str, detailed_logs: List[Dict], result: Dict):
        """Save detailed results for a language pair"""
        output_file = os.path.join(self.config.output_dir, f"{pair_name}_detailed_results.json")
        
        try:
            # Prepare comprehensive output
            output_data = {
                "evaluation_metadata": {
                    "pair_name": pair_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model_name": self.config.model_name,
                    "translation_direction": self.config.translation_direction,
                    "num_samples": len(detailed_logs),
                    "config": {
                        "batch_size": self.config.batch_size,
                        "max_new_tokens": self.config.max_new_tokens,
                        "use_few_shot": self.config.use_few_shot,
                        "num_few_shot_examples": self.config.num_few_shot_examples
                    }
                },
                "results": result,
                "detailed_logs": detailed_logs
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Saved detailed results to {output_file}")
            
        except Exception as e:
            print(f"  ❌ Error saving detailed results: {e}")
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation across all language pairs"""
        if self.accelerator.is_main_process:
            print("="*70)
            print("🚀 Starting Enhanced FLORES Translation Evaluation")
            print(f"Model: {self.config.model_name}")
            print(f"Direction: {self.config.translation_direction}")
            print(f"Languages: {', '.join(self.config.target_languages)}")
            print(f"Samples per language: {self.config.num_samples_per_lang}")
            print("="*70)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Initialize results storage
        overall_scores = {}
        all_chrf_scores = []
        all_detailed_logs = []
        
        # Evaluate each language pair
        lang_iter = tqdm(
            self.config.target_languages,
            desc="Evaluating languages",
            disable=not self.accelerator.is_main_process
        )
        
        for indic_lang_code in lang_iter:
            source_lang, target_lang = self._get_language_pair(indic_lang_code)
            pair_name = f"flores_{source_lang}-{target_lang}"
            lang_iter.set_postfix({"current": pair_name})
            
            try:
                result, detailed_logs = self.evaluate_language_pair(
                    model, tokenizer, indic_lang_code
                )
                
                if result is not None:
                    overall_scores[pair_name] = result
                    
                    if "chrf++" in result and isinstance(result["chrf++"], (int, float)):
                        all_chrf_scores.append(result["chrf++"])
                    
                    if detailed_logs:
                        all_detailed_logs.extend(detailed_logs)
                
                # Synchronize across processes
                self.accelerator.wait_for_everyone()
                
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"❌ Failed to evaluate {pair_name}: {e}")
                    traceback.print_exc()
                continue
        
        # Save comprehensive results
        if self.accelerator.is_main_process:
            self._save_comprehensive_results(overall_scores, all_detailed_logs, all_chrf_scores)
            self._print_final_summary(overall_scores, all_chrf_scores)
        
        return overall_scores
    
    def _save_comprehensive_results(self, overall_scores: Dict, all_detailed_logs: List[Dict], all_chrf_scores: List[float]):
        """Save comprehensive evaluation results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save overall summary
        summary_file = os.path.join(
            self.config.output_dir,
            f"flores_evaluation_summary_{timestamp}.json"
        )
        
        summary_data = {
            "evaluation_metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": self.config.model_name,
                "translation_direction": self.config.translation_direction,
                "total_language_pairs": len(overall_scores),
                "total_samples": len(all_detailed_logs),
                "config": self.config.__dict__
            },
            "overall_results": {
                "average_chrf": np.mean(all_chrf_scores) if all_chrf_scores else 0.0,
                "std_chrf": np.std(all_chrf_scores) if all_chrf_scores else 0.0,
                "min_chrf": np.min(all_chrf_scores) if all_chrf_scores else 0.0,
                "max_chrf": np.max(all_chrf_scores) if all_chrf_scores else 0.0,
                "median_chrf": np.median(all_chrf_scores) if all_chrf_scores else 0.0
            },
            "language_pair_results": overall_scores
        }
        
        try:
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Comprehensive summary saved to: {summary_file}")
        except Exception as e:
            print(f"❌ Error saving comprehensive summary: {e}")
        
        # Save all detailed logs
        detailed_file = os.path.join(
            self.config.output_dir,
            f"flores_all_detailed_logs_{timestamp}.json"
        )
        
        try:
            with open(detailed_file, "w", encoding="utf-8") as f:
                json.dump(all_detailed_logs, f, indent=2, ensure_ascii=False)
            print(f"💾 All detailed logs saved to: {detailed_file}")
        except Exception as e:
            print(f"❌ Error saving detailed logs: {e}")
    
    def _print_final_summary(self, overall_scores: Dict, all_chrf_scores: List[float]):
        """Print comprehensive final summary"""
        print("\n" + "="*70)
        print("🏆 FINAL FLORES EVALUATION SUMMARY 🏆")
        print(f"Model: {self.config.model_name}")
        print(f"Direction: {self.config.translation_direction}")
        print("="*70)
        
        # Print individual language pair results
        print("\n📊 Language Pair Results (chrF++):")
        for pair_name, scores in sorted(overall_scores.items()):
            if "error" not in scores:
                chrf_score = scores.get("chrf++", 0.0)
                valid_info = f" ({scores.get('valid_samples', 0)}/{scores.get('total_samples', 0)} valid)" if 'valid_samples' in scores else ""
                validity_rate = scores.get('validity_rate', 0.0) * 100
                print(f"  • {pair_name:20} | chrF++: {chrf_score:6.2f}{valid_info} | Validity: {validity_rate:5.1f}%")
            else:
                print(f"  • {pair_name:20} | ERROR: {scores.get('error', 'Unknown error')}")
        
        # Print overall statistics
        if all_chrf_scores:
            print(f"\n📈 Overall Statistics:")
            print(f"  • Average chrF++:     {np.mean(all_chrf_scores):6.2f}")
            print(f"  • Standard Deviation: {np.std(all_chrf_scores):6.2f}")
            print(f"  • Median chrF++:      {np.median(all_chrf_scores):6.2f}")
            print(f"  • Min chrF++:         {np.min(all_chrf_scores):6.2f}")
            print(f"  • Max chrF++:         {np.max(all_chrf_scores):6.2f}")
            print(f"  • Language Pairs:     {len(all_chrf_scores)}")
            
            # Performance analysis
            avg_score = np.mean(all_chrf_scores)
            if avg_score >= 40.0:
                performance = "🌟 Excellent"
            elif avg_score >= 30.0:
                performance = "✅ Good"
            elif avg_score >= 20.0:
                performance = "⚠️  Fair"
            else:
                performance = "❌ Needs Improvement"
            
            print(f"\n🎯 Performance Assessment: {performance} (Average: {avg_score:.2f})")
            
            # Recommendations
            if avg_score < 30.0:
                print("\n💡 Recommendations for Improvement:")
                print("  • Consider fine-tuning the prompt format")
                print("  • Adjust generation parameters (temperature, top_p)")
                print("  • Increase few-shot examples or improve their quality")
                print("  • Check if the model supports the target languages well")
        else:
            print("\n⚠️ No valid chrF++ scores computed.")
            print("Check the evaluation logs for potential issues.")
        
        print("\n✅ Evaluation Complete!")
        print("="*70)


def main():
    """Main execution function"""
    # Set up memory optimization
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.85, device=i)
    
    # Initialize configuration
    config = FloresConfig()
    
    # Create evaluator and run evaluation
    evaluator = FloresEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()
    
    return results


if __name__ == "__main__":
    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
        results = main()
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"Results summary: {len(results)} language pairs evaluated")
    except Exception as e:
        print(f"\n💥 Evaluation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)