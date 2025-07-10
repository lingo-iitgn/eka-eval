# scripts/run_evaluation_suite.py

import subprocess
import multiprocessing
import os
import pandas as pd 
import time
import sys
import json
import argparse
import logging # For logging levels
from typing import List, Dict as PyDict, Tuple, Set, Any
from collections import defaultdict

import sys
import os

# Set GPU constraint at the very beginning
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""Project path configuration"""
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eka_eval.benchmarks.benchmark_registry import BenchmarkRegistry
from eka_eval.utils.gpu_utils import get_available_gpus
from eka_eval.utils.logging_setup import setup_logging
from eka_eval.utils import constants 
from eka_eval.core.model_loader import get_model_selection_interface
from eka_eval.core.api_loader import get_available_api_models

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Visualization libraries not available: {e}")
    print("Install with: pip install matplotlib seaborn plotly")

csv_path="calculated.csv"

logger = logging.getLogger(__name__) 

def get_constrained_gpus():
    """
    Get available GPUs respecting the CUDA_VISIBLE_DEVICES constraint.
    Returns logical GPU IDs (0, 1) which map to physical GPUs (2, 3).
    """
    # Since we set CUDA_VISIBLE_DEVICES="2,3", PyTorch/CUDA will see:
    # Logical GPU 0 -> Physical GPU 2
    # Logical GPU 1 -> Physical GPU 3
    
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.warning("No GPUs available or CUDA not accessible with current CUDA_VISIBLE_DEVICES setting")
        return []
    
    # Map logical to physical for logging
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    physical_mapping = {}
    for logical_id, physical_id in enumerate(cuda_visible):
        if physical_id.strip().isdigit():
            physical_mapping[logical_id] = int(physical_id.strip())
    
    logger.info(f"GPU constraint active: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"Available logical GPUs: {available_gpus}")
    logger.info(f"Logical -> Physical GPU mapping: {physical_mapping}")
    
    return available_gpus

def worker_process(
    assigned_physical_gpu_id: int,
    subprocess_unique_id: int,
    model_name_or_path: str,
    total_num_workers: int,
    task_group_to_run: str,
    selected_benchmarks_for_group: List[str],
    orchestrator_batch_size: int,
    is_api_model: bool = False,
    api_provider: str = None,
    api_key: str = None,
):
    """
    Manages the execution of a single worker (evaluation_worker.py) as a subprocess.
    Now supports both local and API models with GPU constraints.
    """
    # Ensure GPU constraint is propagated to worker processes
    if not is_api_model:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    
    worker_log_prefix = f"Worker {subprocess_unique_id} ({'API' if is_api_model else f'GPU {assigned_physical_gpu_id}'})"
    model_type_str = f"{api_provider} API" if is_api_model else "Local"
    logger.info(
        f"{worker_log_prefix}: Starting {model_type_str} model '{model_name_or_path}' for TG: '{task_group_to_run}', "
        f"BMs: {selected_benchmarks_for_group}, BatchSize: {orchestrator_batch_size}"
    )
    try:
        python_executable = sys.executable or "python3"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        worker_script_path = os.path.join(project_root, "scripts", "evaluation_worker.py")

        if not os.path.exists(worker_script_path):
            logger.error(f"{worker_log_prefix}: CRITICAL - Worker script not found at {worker_script_path}. Aborting this worker.")
            return

        command = [
            python_executable, "-u", worker_script_path,
            "--gpu_id", str(assigned_physical_gpu_id),
            "--num_gpus", str(total_num_workers),
            "--process_id", str(subprocess_unique_id), 
            "--model_name", model_name_or_path,
            "--batch_size", str(orchestrator_batch_size),
            "--task_group", task_group_to_run,
            "--selected_benchmarks_json", json.dumps({task_group_to_run: selected_benchmarks_for_group}),
            "--results_dir", constants.DEFAULT_RESULTS_DIR if hasattr(constants, 'DEFAULT_RESULTS_DIR') else "results_output"
        ]

        # Add API-specific parameters
        if is_api_model:
            command.extend([
                "--is_api_model", "true",
                "--api_provider", api_provider or "",
                "--api_key", api_key or ""
            ])

        # Set environment for subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "2,3"

        logger.debug(f"{worker_log_prefix}: Executing command: {' '.join(command[:-1])} [API_KEY_HIDDEN]" if is_api_model else f"{worker_log_prefix}: Executing command: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            env=env  # Pass environment with GPU constraint
        )

        logger.info(f"\n--------- Output from {worker_log_prefix} for TG '{task_group_to_run}' ---------\n")
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(f"[{worker_log_prefix}] {line}")
                sys.stdout.flush()
            process.stdout.close()

        return_code = process.wait()
        logger.info(f"\n--------- End Output from {worker_log_prefix} for TG '{task_group_to_run}' ---------")

        if return_code == 0:
            logger.info(f"{worker_log_prefix}: Finished TG '{task_group_to_run}' successfully (RC: {return_code}).")
        else:
            logger.error(f"{worker_log_prefix}: TG '{task_group_to_run}' exited with error (RC: {return_code}).")

    except Exception as e:
        logger.critical(
            f"{worker_log_prefix}: FATAL error launching/monitoring worker for TG '{task_group_to_run}': {e}",
            exc_info=True
        )

def create_visualizations(
    results_csv_path: str,
    model_name: str = None,
    viz_types: List[str] = None,
    output_dir: str = None
):
    """
    Create various visualizations from evaluation results.
    
    Args:
        results_csv_path: Path to the results CSV file
        model_name: Specific model to visualize (None for all models)
        viz_types: List of visualization types to create
        output_dir: Directory to save visualizations
    """
    if not VISUALIZATION_AVAILABLE:
        logger.error("Visualization libraries not available. Please install matplotlib, seaborn, and plotly.")
        return False
    
    if not os.path.exists(results_csv_path):
        logger.error(f"Results file not found: {results_csv_path}")
        return False
    
    # Load data
    try:
        df = pd.read_csv(results_csv_path)
        logger.info(f"Loaded {len(df)} results from {results_csv_path}")
    except Exception as e:
        logger.error(f"Error loading results CSV: {e}")
        return False
    
    # Filter by model if specified
    if model_name:
        df = df[df['Model'] == model_name]
        if df.empty:
            logger.warning(f"No results found for model: {model_name}")
            return False
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(results_csv_path), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Default visualization types
    if viz_types is None:
        viz_types = ["heatmap", "bar_chart", "radar_chart", "model_comparison", "task_breakdown"]
    
    success = True
    
    try:
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        for viz_type in viz_types:
            try:
                if viz_type == "heatmap":
                    success &= create_heatmap(df, output_dir, model_name)
                elif viz_type == "bar_chart":
                    success &= create_bar_chart(df, output_dir, model_name)
                elif viz_type == "radar_chart":
                    success &= create_radar_chart(df, output_dir, model_name)
                elif viz_type == "model_comparison":
                    success &= create_model_comparison(df, output_dir)
                elif viz_type == "task_breakdown":
                    success &= create_task_breakdown(df, output_dir, model_name)
                elif viz_type == "interactive_dashboard":
                    success &= create_interactive_dashboard(df, output_dir, model_name)
                else:
                    logger.warning(f"Unknown visualization type: {viz_type}")
            except Exception as e:
                logger.error(f"Error creating {viz_type}: {e}")
                success = False
    
    except Exception as e:
        logger.error(f"Error setting up visualizations: {e}")
        return False
    
    if success:
        logger.info(f"Visualizations saved to: {output_dir}")
    
    return success

def create_heatmap(df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Create a heatmap of scores across tasks and benchmarks"""
    try:
        # Prepare data for heatmap
        pivot_data = df[df['Benchmark'] != 'Average'].pivot_table(
            values='Score', 
            index='Model', 
            columns=['Task', 'Benchmark'], 
            aggfunc='mean'
        )
        
        if pivot_data.empty:
            logger.warning("No data available for heatmap")
            return False
        
        # Create heatmap
        plt.figure(figsize=(16, 8))
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlBu_r',
            center=0.5,
            cbar_kws={'label': 'Score'}
        )
        
        title = f"Performance Heatmap - {model_name}" if model_name else "Performance Heatmap - All Models"
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Task - Benchmark', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = f"heatmap_{model_name.replace('/', '_')}.png" if model_name else "heatmap_all_models.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        return False

def create_bar_chart(df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Create bar charts showing performance across different metrics"""
    try:
        # Average scores by task group
        avg_scores = df[df['Benchmark'] == 'Average'].groupby(['Model', 'Task'])['Score'].mean().reset_index()
        
        if avg_scores.empty:
            logger.warning("No average scores available for bar chart")
            return False
        
        plt.figure(figsize=(14, 8))
        
        if model_name:
            model_data = avg_scores[avg_scores['Model'] == model_name]
            bars = plt.bar(model_data['Task'], model_data['Score'], color='skyblue', alpha=0.8)
            title = f"Average Performance by Task - {model_name}"
        else:
            # Multiple models comparison
            pivot_avg = avg_scores.pivot(index='Task', columns='Model', values='Score')
            pivot_avg.plot(kind='bar', figsize=(14, 8), alpha=0.8)
            title = "Average Performance by Task - Model Comparison"
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Task Group', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        if model_name:
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height):
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        filename = f"bar_chart_{model_name.replace('/', '_')}.png" if model_name else "bar_chart_comparison.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bar chart saved: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating bar chart: {e}")
        return False

def create_radar_chart(df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Create radar chart showing model performance across different dimensions"""
    try:
        import numpy as np
        
        # Get average scores by task
        avg_scores = df[df['Benchmark'] == 'Average'].groupby(['Model', 'Task'])['Score'].mean().reset_index()
        
        if avg_scores.empty:
            logger.warning("No average scores available for radar chart")
            return False
        
        # Prepare data
        if model_name:
            model_data = avg_scores[avg_scores['Model'] == model_name]
            models_to_plot = [model_name]
        else:
            model_data = avg_scores
            models_to_plot = avg_scores['Model'].unique()[:5]  # Limit to 5 models for readability
        
        tasks = avg_scores['Task'].unique()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Set up angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_to_plot)))
        
        for i, model in enumerate(models_to_plot):
            model_scores = []
            for task in tasks:
                score_row = model_data[(model_data['Model'] == model) & (model_data['Task'] == task)]
                score = score_row['Score'].iloc[0] if not score_row.empty else 0
                model_scores.append(score)
            
            model_scores += model_scores[:1]  # Complete the circle
            
            ax.plot(angles, model_scores, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, model_scores, alpha=0.25, color=colors[i])
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        title = f"Performance Radar Chart - {model_name}" if model_name else "Performance Radar Chart - Model Comparison"
        plt.title(title, size=16, fontweight='bold', pad=20)
        
        if len(models_to_plot) > 1:
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        filename = f"radar_chart_{model_name.replace('/', '_')}.png" if model_name else "radar_chart_comparison.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Radar chart saved: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        return False

def create_model_comparison(df: pd.DataFrame, output_dir: str):
    """Create comprehensive model comparison visualizations"""
    try:
        # Overall performance comparison
        avg_scores = df[df['Benchmark'] == 'Average'].groupby('Model')['Score'].mean().sort_values(ascending=False)
        
        if avg_scores.empty:
            logger.warning("No data available for model comparison")
            return False
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(avg_scores)), avg_scores.values, color='lightcoral', alpha=0.8)
        
        plt.title('Overall Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(range(len(avg_scores)), avg_scores.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, avg_scores.values)):
            plt.text(bar.get_x() + bar.get_width()/2., score + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison_overall.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Task-specific comparison
        task_comparison = df[df['Benchmark'] == 'Average'].pivot(index='Task', columns='Model', values='Score')
        
        plt.figure(figsize=(14, 8))
        task_comparison.plot(kind='bar', alpha=0.8)
        plt.title('Model Performance by Task Group', fontsize=16, fontweight='bold')
        plt.xlabel('Task Group', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison_by_task.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison charts saved")
        return True
        
    except Exception as e:
        logger.error(f"Error creating model comparison: {e}")
        return False

def create_task_breakdown(df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Create detailed breakdown of performance within each task"""
    try:
        # Filter data
        plot_df = df[df['Benchmark'] != 'Average'].copy() if not df.empty else pd.DataFrame()
        
        if model_name:
            plot_df = plot_df[plot_df['Model'] == model_name]
        
        if plot_df.empty:
            logger.warning("No detailed benchmark data available for task breakdown")
            return False
        
        # Create subplots for each task
        tasks = plot_df['Task'].unique()
        n_tasks = len(tasks)
        
        if n_tasks == 0:
            return False
        
        fig, axes = plt.subplots(n_tasks, 1, figsize=(12, 4 * n_tasks))
        if n_tasks == 1:
            axes = [axes]
        
        for i, task in enumerate(tasks):
            task_data = plot_df[plot_df['Task'] == task]
            
            if model_name:
                # Single model - show benchmarks
                benchmark_scores = task_data.groupby('Benchmark')['Score'].mean().sort_values(ascending=True)
                bars = axes[i].barh(range(len(benchmark_scores)), benchmark_scores.values, color='lightblue', alpha=0.8)
                axes[i].set_yticks(range(len(benchmark_scores)))
                axes[i].set_yticklabels(benchmark_scores.index)
                axes[i].set_xlabel('Score')
                
                # Add value labels
                for j, (bar, score) in enumerate(zip(bars, benchmark_scores.values)):
                    if pd.notna(score):
                        axes[i].text(score + 0.01, bar.get_y() + bar.get_height()/2.,
                                   f'{score:.3f}', va='center', fontweight='bold')
            else:
                # Multiple models - show comparison
                task_pivot = task_data.pivot_table(values='Score', index='Benchmark', columns='Model', aggfunc='mean')
                task_pivot.plot(kind='bar', ax=axes[i], alpha=0.8)
                axes[i].set_xticklabels(task_pivot.index, rotation=45, ha='right')
                axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            axes[i].set_title(f'Task: {task}', fontweight='bold')
            axes[i].grid(axis='x' if model_name else 'y', alpha=0.3)
        
        title = f"Task Breakdown - {model_name}" if model_name else "Task Breakdown - All Models"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = f"task_breakdown_{model_name.replace('/', '_')}.png" if model_name else "task_breakdown_all.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Task breakdown saved: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating task breakdown: {e}")
        return False

def create_interactive_dashboard(df: pd.DataFrame, output_dir: str, model_name: str = None):
    """Create an interactive Plotly dashboard"""
    try:
        # Overall performance chart
        avg_scores = df[df['Benchmark'] == 'Average'].groupby(['Model', 'Task'])['Score'].mean().reset_index()
        
        if avg_scores.empty:
            logger.warning("No data available for interactive dashboard")
            return False
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Performance', 'Task Comparison', 'Score Distribution', 'Detailed Breakdown'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Overall performance
        if model_name:
            model_avg = avg_scores[avg_scores['Model'] == model_name].groupby('Model')['Score'].mean()
            fig.add_trace(
                go.Bar(x=model_avg.index, y=model_avg.values, name="Overall Score"),
                row=1, col=1
            )
        else:
            overall_avg = avg_scores.groupby('Model')['Score'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=overall_avg.index, y=overall_avg.values, name="Overall Score"),
                row=1, col=1
            )
        
        # 2. Task comparison
        for task in avg_scores['Task'].unique():
            task_data = avg_scores[avg_scores['Task'] == task]
            fig.add_trace(
                go.Bar(x=task_data['Model'], y=task_data['Score'], name=task),
                row=1, col=2
            )
        
        # 3. Score distribution
        all_scores = df[df['Score'].notna()]['Score']
        fig.add_trace(
            go.Histogram(x=all_scores, name="Score Distribution", nbinsx=20),
            row=2, col=1
        )
        
        # 4. Detailed scatter plot
        detailed_df = df[df['Benchmark'] != 'Average'].copy()
        if not detailed_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=detailed_df['Task'],
                    y=detailed_df['Score'],
                    mode='markers',
                    text=detailed_df['Benchmark'],
                    name="Detailed Scores",
                    marker=dict(size=8, opacity=0.7)
                ),
                row=2, col=2
            )
        
        # Update layout
        title = f"Interactive Dashboard - {model_name}" if model_name else "Interactive Dashboard - All Models"
        fig.update_layout(
            title_text=title,
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Save interactive HTML
        filename = f"dashboard_{model_name.replace('/', '_')}.html" if model_name else "dashboard_all.html"
        fig.write_html(os.path.join(output_dir, filename))
        
        logger.info(f"Interactive dashboard saved: {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}")
        return False

def main_orchestrator():
    """
    Main function to orchestrate the LLM benchmark evaluations.
    Handles user input, task scheduling, and worker management for both local and API models.
    Updated to respect GPU constraints (GPUs 2 and 3 only).
    """
    parser = argparse.ArgumentParser(description="Eka-Eval: LLM Benchmark Evaluation Suite.")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs/workers to use. Default: all available or 1 for CPU/API.")
    parser.add_argument("--batch_size", type=int, default=1, help="Default batch size for worker tasks.")
    parser.add_argument("--results_dir", type=str, default=constants.DEFAULT_RESULTS_DIR if hasattr(constants, 'DEFAULT_RESULTS_DIR') else "results_output", help="Directory to save evaluation results.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level.")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Create visualizations after evaluation")
    parser.add_argument("--viz_only", action="store_true", help="Only create visualizations (skip evaluation)")
    parser.add_argument("--viz_types", type=str, nargs="+", 
                       choices=["heatmap", "bar_chart", "radar_chart", "model_comparison", "task_breakdown", "interactive_dashboard"],
                       default=["heatmap", "bar_chart", "model_comparison"],
                       help="Types of visualizations to create")
    parser.add_argument("--viz_model", type=str, help="Specific model to visualize (if not specified, all models)")
    parser.add_argument("--viz_output", type=str, help="Output directory for visualizations")
    
    # Decontamination options
    parser.add_argument("--enable_decontamination", action="store_true", help="Enable decontamination checking")
    parser.add_argument("--contamination_report", action="store_true", help="Generate contamination analysis report")

    args = parser.parse_args()
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    setup_logging(level=log_level_map.get(args.log_level.upper(), logging.INFO), worker_id="Orchestrator")

    logger.info("--- Eka-Eval Orchestrator Starting ---")

    # Handle visualization-only mode
    results_csv_path = os.path.join(args.results_dir, 'calculated.csv')
    
    if args.viz_only:
        logger.info("Running in visualization-only mode")
        success = create_visualizations(
            results_csv_path=results_csv_path,
            model_name=args.viz_model,
            viz_types=args.viz_types,
            output_dir=args.viz_output
        )
        if success:
            logger.info("Visualizations completed successfully")
        else:
            logger.error("Visualization creation failed")
        return

    benchmark_registry = BenchmarkRegistry()
    if not benchmark_registry.benchmarks:
        logger.critical("Benchmark configuration is empty or failed to load. Exiting.")
        return

    # --- Model Selection (Updated to support API models) ---
    logger.info("\n--- Model Selection ---")
    try:
        model_info, is_api_model = get_model_selection_interface()
        
        if is_api_model:
            input_model_name = model_info["model_name"]
            api_provider = model_info["provider"]
            api_key = model_info["api_key"]
            logger.info(f"Selected API model: {api_provider}/{input_model_name}")
        else:
            input_model_name = model_info["model_name"]
            api_provider = None
            api_key = None
            logger.info(f"Selected local model: {input_model_name}")
            
            # Validate local model path
            if not input_model_name.startswith('/') and '/' not in input_model_name:
                # Assume it's a Hugging Face model
                pass
            elif os.path.exists(input_model_name) and not os.path.isdir(input_model_name):
                logger.warning(f"Local path '{input_model_name}' does not seem to be a valid directory.")

    except Exception as e:
        logger.error(f"Error in model selection: {e}")
        return

    if not input_model_name:
        logger.error("Model name/path cannot be empty. Exiting.")
        return

    model_name_lower = input_model_name.lower()

    # Custom benchmarks section (unchanged)
    add_custom = input("Do you want to add any custom/internal benchmarks for this session? (yes/no): ").strip().lower()
    if add_custom == 'yes':
        while True:
            logger.info("\n--- Adding Custom Benchmark ---")
            custom_task_group = input("Enter Task Group name for this custom benchmark (e.g., CUSTOM EVALS): ").strip()
            custom_bm_name = input("Enter a unique display name for this custom benchmark (e.g., MySpecialTest): ").strip()
            custom_module_path = input("Enter Python module path (e.g., my_custom_evals.script_name): ").strip()
            custom_func_name = input(f"Enter function name in '{custom_module_path}' to call (e.g., evaluate_my_test): ").strip()

            if not all([custom_task_group, custom_bm_name, custom_module_path, custom_func_name]):
                logger.warning("All fields are required for a custom benchmark. Skipping this one.")
            else:
                success = benchmark_registry.add_custom_benchmark_definition(
                    custom_task_group, custom_bm_name,
                    custom_module_path, custom_func_name,
                    description=f"Custom benchmark: {custom_bm_name}"
                )
                if success:
                    logger.info(f"Added custom benchmark '{custom_bm_name}' under task group '{custom_task_group}'.")

            if input("Add another custom benchmark? (yes/no): ").strip().lower() != 'yes':
                break

    # Task group selection (unchanged)
    logger.info("\n--- Available Benchmark Task Groups ---")
    all_task_groups = benchmark_registry.get_task_groups()
    for i, tg_name in enumerate(all_task_groups):
        print(f"{i+1}. {tg_name}")
    print(f"{len(all_task_groups)+1}. ALL Task Groups")

    selected_indices_str = input(f"Select task group #(s) (e.g., '1', '1 3', 'ALL'): ").strip().lower().split()
    chosen_initial_task_groups: List[str] = []
    if "all" in selected_indices_str or str(len(all_task_groups) + 1) in selected_indices_str:
        chosen_initial_task_groups = all_task_groups
    else:
        for idx_str in selected_indices_str:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(all_task_groups):
                    chosen_initial_task_groups.append(all_task_groups[idx])
                else:
                    logger.warning(f"Invalid task group # '{idx_str}' ignored.")
            except ValueError:
                logger.warning(f"Invalid input '{idx_str}' for task group selection ignored.")

    if not chosen_initial_task_groups:
        logger.error("No valid top-level task groups selected. Exiting.")
        return
    logger.info(f"Selected top-level task groups for further benchmark selection: {chosen_initial_task_groups}")

    # Benchmark selection within task groups (unchanged)
    user_selected_benchmarks: PyDict[str, List[str]] = {}
    ordered_selected_task_groups_for_processing: List[str] = []

    for task_group_name in chosen_initial_task_groups:
        group_benchmarks = benchmark_registry.get_benchmarks_for_group(task_group_name)
        is_single_bm_task_group = len(group_benchmarks) == 1 and group_benchmarks[0] == task_group_name

        if is_single_bm_task_group:
            user_selected_benchmarks[task_group_name] = [task_group_name]
            if task_group_name not in ordered_selected_task_groups_for_processing:
                ordered_selected_task_groups_for_processing.append(task_group_name)
        else:
            logger.info(f"\n--- Select benchmarks for Task Group: {task_group_name} ---")
            available_sub_benchmarks = group_benchmarks
            if not available_sub_benchmarks:
                logger.warning(f"No sub-benchmarks defined for task group '{task_group_name}'. Skipping.")
                continue

            for i, sub_bm in enumerate(available_sub_benchmarks):
                print(f"{i+1}. {sub_bm}")
            print(f"{len(available_sub_benchmarks)+1}. ALL (within {task_group_name})")
            print(f"{len(available_sub_benchmarks)+2}. SKIP THIS TASK GROUP")

            selected_sub_indices_str = input(f"Select benchmark #(s) for {task_group_name} ('ALL', 'SKIP', nums): ").strip().lower().split()
            selected_for_this_group: List[str] = []

            if "skip" in selected_sub_indices_str or str(len(available_sub_benchmarks)+2) in selected_sub_indices_str:
                logger.info(f"Skipping task group: {task_group_name}")
                continue
            if "all" in selected_sub_indices_str or str(len(available_sub_benchmarks)+1) in selected_sub_indices_str:
                selected_for_this_group = available_sub_benchmarks
            else:
                for sub_idx_str in selected_sub_indices_str:
                    try:
                        sub_idx = int(sub_idx_str) - 1
                        if 0 <= sub_idx < len(available_sub_benchmarks):
                            selected_for_this_group.append(available_sub_benchmarks[sub_idx])
                        else:
                            logger.warning(f"Invalid benchmark # '{sub_idx_str}' for {task_group_name} ignored.")
                    except ValueError:
                        logger.warning(f"Invalid input '{sub_idx_str}' for {task_group_name} ignored.")

            if selected_for_this_group:
                user_selected_benchmarks[task_group_name] = sorted(list(set(selected_for_this_group)))
                if task_group_name not in ordered_selected_task_groups_for_processing:
                    ordered_selected_task_groups_for_processing.append(task_group_name)

    if not user_selected_benchmarks:
        logger.info("No benchmarks were selected for evaluation. Exiting.")
        return

    logger.info("\n--- Final Benchmarks Selected for Evaluation (Task Group: [Specific Benchmarks]) ---")
    for tg_name in ordered_selected_task_groups_for_processing:
        if tg_name in user_selected_benchmarks:
            logger.info(f"- {tg_name}: {user_selected_benchmarks[tg_name]}")

    # Check for completed benchmarks (unchanged)
    completed_benchmarks_set: Set[Tuple[str, str]] = set()
    if os.path.exists(results_csv_path):
        try:
            df = pd.read_csv(results_csv_path)
            if all(col in df.columns for col in ['Model', 'Task', 'Benchmark', 'Score']):
                model_df = df[df['Model'] == input_model_name] 
                for _, row in model_df.iterrows():
                    if pd.notna(row['Score']):
                        completed_benchmarks_set.add((row['Task'], row['Benchmark']))
                logger.info(f"Found {len(completed_benchmarks_set)} completed benchmarks for '{input_model_name}' in '{results_csv_path}'.")
            else:
                logger.warning(f"Results file '{results_csv_path}' is missing required columns. Assuming no completed benchmarks.")
        except Exception as e:
            logger.error(f"Error loading completed benchmarks from '{results_csv_path}': {e}. Assuming no completed benchmarks.", exc_info=True)

    # Filter tasks needing evaluation (unchanged)
    tasks_to_schedule_for_workers: PyDict[str, List[str]] = defaultdict(list)
    for task_group, selected_bms_for_group in user_selected_benchmarks.items():
        bms_needing_eval_for_group = [
            bm for bm in selected_bms_for_group if (task_group, bm) not in completed_benchmarks_set
        ]
        if bms_needing_eval_for_group:
            tasks_to_schedule_for_workers[task_group] = bms_needing_eval_for_group

    if not tasks_to_schedule_for_workers:
        logger.info(f"All specifically selected benchmarks for model '{input_model_name}' are already completed and found in '{results_csv_path}'.")
        display_consolidated_results(input_model_name, results_csv_path, user_selected_benchmarks, ordered_selected_task_groups_for_processing, benchmark_registry)
        
        # Ask about visualizations for completed results
        if not args.visualize and not args.viz_only:
            viz_prompt = input("\nAll selected benchmarks are already completed. Do you want to create visualizations for the existing results? (yes/no): ").strip().lower()
            should_create_viz = viz_prompt in ['yes', 'y', '1', 'true']
        else:
            should_create_viz = args.visualize
        
        # Create visualizations if requested
        if should_create_viz:
            logger.info("Creating visualizations for completed results...")
            
            # Interactive configuration for visualizations
            viz_types_to_use = args.viz_types
            viz_output_to_use = args.viz_output
            viz_model_to_use = input_model_name
            
            if not args.visualize and not args.viz_only:
                logger.info("\n--- Visualization Configuration ---")
                
                # Ask about visualization types
                print("Available visualization types:")
                available_viz_types = ["heatmap", "bar_chart", "radar_chart", "model_comparison", "task_breakdown", "interactive_dashboard"]
                for i, viz_type in enumerate(available_viz_types, 1):
                    print(f"{i}. {viz_type}")
                print(f"{len(available_viz_types) + 1}. ALL (create all types)")
                
                viz_selection = input(f"Select visualization types (e.g., '1 3 5' or 'ALL'): ").strip().lower()
                
                if viz_selection == 'all' or viz_selection == str(len(available_viz_types) + 1):
                    viz_types_to_use = available_viz_types
                else:
                    selected_indices = []
                    for idx_str in viz_selection.split():
                        try:
                            idx = int(idx_str) - 1
                            if 0 <= idx < len(available_viz_types):
                                selected_indices.append(idx)
                        except ValueError:
                            pass
                    
                    if selected_indices:
                        viz_types_to_use = [available_viz_types[i] for i in selected_indices]
                    else:
                        logger.warning("No valid visualization types selected. Using default types.")
                        viz_types_to_use = ["heatmap", "bar_chart", "model_comparison"]
                
                # Ask about model filter
                model_filter_choice = input(f"Create visualizations for current model only ({input_model_name}) or all models? (current/all): ").strip().lower()
                if model_filter_choice in ['all', 'a']:
                    viz_model_to_use = None
                
                # Ask about output directory
                custom_output = input("Custom output directory for visualizations (press Enter for default): ").strip()
                if custom_output:
                    viz_output_to_use = custom_output
            
            success = create_visualizations(
                results_csv_path=results_csv_path,
                model_name=viz_model_to_use,
                viz_types=viz_types_to_use,
                output_dir=viz_output_to_use
            )
            
            if success:
                logger.info("✅ Visualizations created successfully!")
            else:
               logger.info("✅ Visualizations created successfully!")
        
        return

    logger.info(f"\n--- Tasks Requiring Evaluation (Task Group: [Benchmarks]) ---")
    for tg_name in ordered_selected_task_groups_for_processing:
        if tg_name in tasks_to_schedule_for_workers:
            logger.info(f"- {tg_name}: {tasks_to_schedule_for_workers[tg_name]}")

    # Worker configuration (Updated for API models and GPU constraints)
    if is_api_model:
        # API models don't need GPU workers
        total_workers_to_use = 1  # Start with single worker for API models
        effective_gpu_ids_for_assignment = [-1]  # -1 indicates API mode
        logger.info(f"Using API model ({api_provider}). Running with 1 worker in API mode.")
        
        # For API models, we could potentially allow multiple workers if rate limits permit
        if args.num_gpus and args.num_gpus > 1:
            logger.info(f"Note: You specified {args.num_gpus} workers, but API models typically work better with fewer concurrent requests due to rate limits.")
            confirm = input(f"Do you want to use {args.num_gpus} concurrent API workers? (y/n): ").strip().lower()
            if confirm == 'y':
                total_workers_to_use = args.num_gpus
                effective_gpu_ids_for_assignment = [-1] * total_workers_to_use
                logger.info(f"Using {total_workers_to_use} concurrent API workers. Monitor for rate limiting.")
    else:
        # Local model logic (Updated with GPU constraints)
        available_physical_gpu_ids = get_constrained_gpus()  # Use our constrained GPU function
        is_cpu_run = not available_physical_gpu_ids 
        if is_cpu_run:
            total_workers_to_use = 1
            effective_gpu_ids_for_assignment = [-1]
            logger.info("No GPUs found or CUDA not available with current constraints. Running in CPU mode with 1 worker.")
        else: 
            num_available_gpu_slots = len(available_physical_gpu_ids)
            if args.num_gpus is not None and args.num_gpus > 0:
                if args.num_gpus > num_available_gpu_slots:
                    logger.warning(
                        f"Requested {args.num_gpus} GPUs, but only {num_available_gpu_slots} are available (GPUs 2,3). "
                        f"Using {num_available_gpu_slots} worker(s)."
                    )
                    total_workers_to_use = num_available_gpu_slots
                else:
                    total_workers_to_use = args.num_gpus
            else:
                total_workers_to_use = num_available_gpu_slots
            effective_gpu_ids_for_assignment = available_physical_gpu_ids[:total_workers_to_use]
            logger.info(f"Using {total_workers_to_use} GPU worker(s) targeting constrained GPUs: {effective_gpu_ids_for_assignment} (Physical GPUs 2,3).")

    if total_workers_to_use == 0:
        logger.error("Error: No workers available. Exiting.")
        return

    # Prepare work items (unchanged)
    work_items_to_distribute: List[PyDict[str, Any]] = []
    for tg_name_ordered in ordered_selected_task_groups_for_processing:
        if tg_name_ordered in tasks_to_schedule_for_workers:
            work_items_to_distribute.append({
                'task_group': tg_name_ordered,
                'benchmarks': tasks_to_schedule_for_workers[tg_name_ordered]
            })

    # Launch worker processes (Updated with API support and GPU constraints)
    processes = []
    worker_type_str = "API" if is_api_model else "GPU/CPU"
    logger.info(f"\n--- Launching {len(work_items_to_distribute)} evaluation tasks across {total_workers_to_use} {worker_type_str} worker(s) ---")

    for i, work_item in enumerate(work_items_to_distribute):
        worker_slot_index = i % total_workers_to_use
        assigned_physical_gpu_id = effective_gpu_ids_for_assignment[worker_slot_index]

        task_group_to_run = work_item['task_group']
        specific_benchmarks_for_group = work_item['benchmarks']
        subprocess_unique_id = i 

        if is_api_model:
            worker_target_str = "API"
        else:
            # For constrained GPUs, show both logical and physical mapping
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if assigned_physical_gpu_id >= 0 and assigned_physical_gpu_id < len(cuda_visible):
                physical_gpu = cuda_visible[assigned_physical_gpu_id]
                worker_target_str = f"Logical GPU {assigned_physical_gpu_id} (Physical GPU {physical_gpu})"
            else:
                worker_target_str = "CPU" if assigned_physical_gpu_id == -1 else f"GPU {assigned_physical_gpu_id}"
        
        logger.info(
            f"Preparing Subprocess {subprocess_unique_id} (mapped to worker slot {worker_slot_index}, "
            f"targeting {worker_target_str}): "
            f"TG '{task_group_to_run}', BMs: {specific_benchmarks_for_group}"
        )
        
        p = multiprocessing.Process(
            target=worker_process,
            args=(
                assigned_physical_gpu_id,
                subprocess_unique_id,
                input_model_name,
                total_workers_to_use,
                task_group_to_run,
                specific_benchmarks_for_group,
                args.batch_size,
                is_api_model,
                api_provider,
                api_key
            )
        )
        processes.append(p)
        p.start()

        # Add delay for API models to prevent rate limiting
        if is_api_model and len(work_items_to_distribute) > 1:
            time.sleep(2)  # Longer delay for API models
        elif total_workers_to_use > 1 and not is_api_model and len(processes) >= total_workers_to_use:
            time.sleep(max(1, 3 // total_workers_to_use if total_workers_to_use > 0 else 3))

    logger.info(f"All {len(processes)} worker processes launched. Waiting for completion...")
    for i, p in enumerate(processes):
        p.join()
        logger.info(f"Worker process {i} (Subprocess UID {i}) has finished.")

    logger.info("\n--- All evaluation worker processes have completed. ---")
    logger.info("Consolidating and displaying results...")
    display_consolidated_results(
        input_model_name,
        results_csv_path,
        user_selected_benchmarks,
        ordered_selected_task_groups_for_processing,
        benchmark_registry 
    )
    
    # Ask about visualizations interactively
    should_create_visualizations = args.visualize
    
    if not should_create_visualizations and not args.viz_only:
        viz_prompt = input("\nDo you want to create visualizations for the results? (yes/no): ").strip().lower()
        should_create_visualizations = viz_prompt in ['yes', 'y', '1', 'true']
    
    if should_create_visualizations:
        # Interactive visualization configuration
        viz_types_to_create = args.viz_types
        viz_output_dir = args.viz_output
        viz_model_filter = args.viz_model or input_model_name
        
        if not args.visualize and not args.viz_only:  # Only ask if not specified via CLI
            logger.info("\n--- Visualization Configuration ---")
            
            # Ask about visualization types
            print("Available visualization types:")
            available_viz_types = ["heatmap", "bar_chart", "radar_chart", "model_comparison", "task_breakdown", "interactive_dashboard"]
            for i, viz_type in enumerate(available_viz_types, 1):
                print(f"{i}. {viz_type}")
            print(f"{len(available_viz_types) + 1}. ALL (create all types)")
            
            viz_selection = input(f"Select visualization types (e.g., '1 3 5' or 'ALL'): ").strip().lower()
            
            if viz_selection == 'all' or viz_selection == str(len(available_viz_types) + 1):
                viz_types_to_create = available_viz_types
            else:
                selected_indices = []
                for idx_str in viz_selection.split():
                    try:
                        idx = int(idx_str) - 1
                        if 0 <= idx < len(available_viz_types):
                            selected_indices.append(idx)
                    except ValueError:
                        pass
                
                if selected_indices:
                    viz_types_to_create = [available_viz_types[i] for i in selected_indices]
                else:
                    logger.warning("No valid visualization types selected. Using default types.")
                    viz_types_to_create = ["heatmap", "bar_chart", "model_comparison"]
            
            # Ask about model filter
            model_filter_choice = input(f"Create visualizations for current model only ({input_model_name}) or all models? (current/all): ").strip().lower()
            if model_filter_choice in ['all', 'a']:
                viz_model_filter = None
            
            # Ask about output directory
            custom_output = input("Custom output directory for visualizations (press Enter for default): ").strip()
            if custom_output:
                viz_output_dir = custom_output
        
        logger.info(f"\n--- Creating Visualizations ---")
        logger.info(f"Types: {viz_types_to_create}")
        logger.info(f"Model filter: {viz_model_filter or 'All models'}")
        logger.info(f"Output directory: {viz_output_dir or 'Default (results_output/visualizations)'}")
        
        success = create_visualizations(
            results_csv_path=results_csv_path,
            model_name=viz_model_filter,
            viz_types=viz_types_to_create,
            output_dir=viz_output_dir
        )
        
        if success:
            logger.info("✅ Visualizations created successfully!")
            if viz_output_dir:
                logger.info(f"📁 Saved to: {viz_output_dir}")
            else:
                default_viz_dir = os.path.join(args.results_dir, "visualizations")
                logger.info(f"📁 Saved to: {default_viz_dir}")
        else:
            logger.warning("⚠️ Some visualizations may have failed. Check logs for details.")
    else:
        logger.info("Skipping visualization creation.")

def display_consolidated_results(
    model_name_to_display: str,
    csv_path: str,
    user_selected_benchmarks_map: PyDict[str, List[str]], 
    ordered_task_groups_for_display: List[str],
    registry: BenchmarkRegistry 
):
    """Displaying consolidated results (unchanged from original)"""
    if not os.path.exists(csv_path):
        logger.error(f"Results file '{csv_path}' not found. Cannot display results.")
        return
    try:
        final_df = pd.read_csv(csv_path)
        model_df_display = final_df[final_df['Model'] == model_name_to_display].copy()

        if model_df_display.empty:
            logger.info(f"\nNo results found for model '{model_name_to_display}' in '{csv_path}'.")
            return

        model_df_display['Score'] = pd.to_numeric(model_df_display['Score'], errors='coerce')

        size_b_val = 'N/A'
        if 'Size (B)' in model_df_display.columns and not model_df_display['Size (B)'].dropna().empty:
            size_b_val = model_df_display['Size (B)'].dropna().iloc[0]

        current_model_row_data = {('Model', ''): model_name_to_display, ('Size (B)', ''): size_b_val}
        task_bm_scores_from_csv = defaultdict(lambda: defaultdict(lambda: pd.NA))
        for _, row in model_df_display.iterrows():
            task_bm_scores_from_csv[row['Task']][row['Benchmark']] = row['Score']

        multi_index_columns_for_df = [('Model', ''), ('Size (B)', '')]

        for task_group_name in ordered_task_groups_for_display:
            if task_group_name not in user_selected_benchmarks_map:
                continue

            selected_bms_in_this_group = user_selected_benchmarks_map.get(task_group_name, [])
            if not selected_bms_in_this_group:
                continue

            registry_benchmarks_for_group = registry.get_benchmarks_for_group(task_group_name)
            is_single_bm_task_group = len(registry_benchmarks_for_group) == 1 and registry_benchmarks_for_group[0] == task_group_name

            if is_single_bm_task_group:
                if task_group_name in selected_bms_in_this_group: 
                    score = task_bm_scores_from_csv[task_group_name].get(task_group_name, pd.NA)
                    current_model_row_data[(task_group_name, '')] = round(score, 2) if pd.notna(score) else pd.NA 
                    multi_index_columns_for_df.append((task_group_name, '')) 
                    current_model_row_data[(task_group_name, 'Average')] = round(score, 2) if pd.notna(score) else pd.NA
                    multi_index_columns_for_df.append((task_group_name, 'Average'))
            else: 
                actual_scores_for_group_avg = []
                canonical_bms_in_group = registry.get_benchmarks_for_group(task_group_name)

                for bm_name in canonical_bms_in_group:
                    if bm_name in selected_bms_in_this_group:
                        score = task_bm_scores_from_csv[task_group_name].get(bm_name, pd.NA)
                        current_model_row_data[(task_group_name, bm_name)] = round(score, 2) if pd.notna(score) else pd.NA
                        multi_index_columns_for_df.append((task_group_name, bm_name))
                        if pd.notna(score):
                            actual_scores_for_group_avg.append(score)

                if len([bm for bm in canonical_bms_in_group if bm in selected_bms_in_this_group]) > 1:
                    avg_score_from_csv = task_bm_scores_from_csv[task_group_name].get('Average', pd.NA)
                    if pd.notna(avg_score_from_csv):
                        current_model_row_data[(task_group_name, 'Average')] = round(avg_score_from_csv, 2)
                    elif actual_scores_for_group_avg:
                        avg_score_calculated = sum(actual_scores_for_group_avg) / len(actual_scores_for_group_avg)
                        current_model_row_data[(task_group_name, 'Average')] = round(avg_score_calculated, 2)
                        logger.warning(f"Calculated average for {task_group_name} as it was missing from CSV.")
                    else:
                        current_model_row_data[(task_group_name, 'Average')] = pd.NA
                    multi_index_columns_for_df.append((task_group_name, 'Average'))

        seen_cols, unique_multi_index_cols = set(), []
        for col_tuple in multi_index_columns_for_df:
            if col_tuple not in seen_cols:
                unique_multi_index_cols.append(col_tuple)
                seen_cols.add(col_tuple)

        if not unique_multi_index_cols or (len(unique_multi_index_cols) == 2 and unique_multi_index_cols[0][0] == 'Model' and unique_multi_index_cols[1][0] == 'Size (B)' ):
             logger.warning("No benchmark score columns to display in table for the selected model.")
             if ('Model', '') in current_model_row_data :
                 print(f"\nModel: {current_model_row_data[('Model', '')]}, Size (B): {current_model_row_data.get(('Size (B)', ''), 'N/A')}")
                 print("No benchmark scores were found or selected for display for this model.")
             return

        df_for_display = pd.DataFrame(columns=pd.MultiIndex.from_tuples(unique_multi_index_cols))
        row_data_for_series = {col_t: current_model_row_data.get(col_t, pd.NA) for col_t in unique_multi_index_cols}
        series_for_df_row = pd.Series(row_data_for_series, index=df_for_display.columns)

        if not series_for_df_row.empty:
            df_for_display.loc[0] = series_for_df_row
        elif unique_multi_index_cols:
            df_for_display.loc[0] = pd.NA

        def sort_key_for_display_table(col_tuple: Tuple[str, str]):
            tg_name, bm_name = col_tuple[0], col_tuple[1]
            if tg_name == 'Model': return (0, 0, 0)
            if tg_name == 'Size (B)': return (1, 0, 0)

            try: 
                task_order_idx = ordered_task_groups_for_display.index(tg_name)
            except ValueError:
                task_order_idx = 9999 
                
            registry_bms_for_group_sort = registry.get_benchmarks_for_group(tg_name)
            is_single_bm_group_sort = len(registry_bms_for_group_sort) == 1 and registry_bms_for_group_sort[0] == tg_name

            if is_single_bm_group_sort:
                sub_order = 0 if bm_name == '' else 1 
                return (2, task_order_idx, sub_order) 
            else: # Multi-benchmark group
                if bm_name == 'Average':
                    bm_order_idx = 99999 # Average last within its group
                else:
                    try:
                        canonical_bms_in_group_sort = registry.get_benchmarks_for_group(tg_name)
                        bm_order_idx = canonical_bms_in_group_sort.index(bm_name)
                    except (KeyError, ValueError):
                        bm_order_idx = 99998
                return (3, task_order_idx, bm_order_idx) 

        cols_to_sort = [col for col in df_for_display.columns.tolist() if col in unique_multi_index_cols]
        if not cols_to_sort and not df_for_display.empty:
             cols_to_sort = df_for_display.columns.tolist()

        if cols_to_sort:
            sorted_display_columns = sorted(cols_to_sort, key=sort_key_for_display_table)
            df_for_display = df_for_display[sorted_display_columns]
        elif df_for_display.empty:
            logger.info("No results data to display in table.")
            return

        logger.info("\n--- Consolidated Evaluation Results ---")
        print(df_for_display.to_markdown(index=False, floatfmt=".2f"))

    except FileNotFoundError:
        logger.error(f"Results file '{csv_path}' not found for display.")
    except Exception as e:
        logger.error(f"Error displaying consolidated results from '{csv_path}': {e}", exc_info=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main_orchestrator()