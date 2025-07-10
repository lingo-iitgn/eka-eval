import pandas as pd

def create_calculated_csv():
    """
    Creates a CSV file named 'calculated.csv' with pre-calculated scores for Falcon 7B.
    """
    data = []

    model_name = 'tiiuae/falcon-7b'
    model_size = 7 # In Billions

    # MATH
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'MATH', 'Benchmark': 'GSM8K', 'Score': 3.185})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'MATH', 'Benchmark': 'MATH', 'Score': 1.4})

    # CODE GENERATION
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'CODE GENERATION', 'Benchmark': 'HumanEval', 'Score': 0.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'CODE GENERATION', 'Benchmark': 'MBPP', 'Score': 12.8})

    # COMMONSENSE REASONING
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'PIQA', 'Score': 49.5})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'SIQA', 'Score': 42.18})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'WinoGrande', 'Score': 62.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'OpenBookQA', 'Score': 54.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'ARC (easy + challenge)', 'Score': 48.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'HellaSwag', 'Score': 44.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'COMMONSENSE REASONING', 'Benchmark': 'CommonsenseQA', 'Score': 41.0})

    # AGGREGATED BENCHMARKS
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'AGGREGATED BENCHMARKS', 'Benchmark': 'MMLU', 'Score': 27.15})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'AGGREGATED BENCHMARKS', 'Benchmark': 'AGIEval', 'Score': 20.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'AGGREGATED BENCHMARKS', 'Benchmark': 'BBH', 'Score': 28.0})

    # READING COMPREHENSION
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'READING COMPREHENSION', 'Benchmark': 'SQuAD', 'Score': 13.25})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'READING COMPREHENSION', 'Benchmark': 'BoolQ', 'Score': 17.0})
    data.append({'Model': model_name, 'Size (B)': model_size, 'Task': 'READING COMPREHENSION', 'Benchmark': 'QuAC', 'Score': 15.15})

    df = pd.DataFrame(data)
    df.to_csv('calculated.csv', index=False)
    print("Successfully created 'calculated.csv' with Falcon 7B pre-calculated results.")

if __name__ == "__main__":
    create_calculated_csv()