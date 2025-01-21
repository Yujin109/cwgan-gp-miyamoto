import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation_metrics(csv_path, output_dir):
    """
    Generate scatter plots for evaluation metrics from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing evaluation results.
        output_dir (str): Directory to save the output plots.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Exclude rows with 'Overall' or empty values
    df = df.dropna()
    df = df[~df["Label"].str.contains("Overall", na=False)]

    # Convert columns to numeric as needed
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df["Convergence Rate"] = pd.to_numeric(df["Convergence Rate"], errors="coerce")
    df["MSE"] = pd.to_numeric(df["MSE"], errors="coerce")
    df["Diversity L2"] = pd.to_numeric(df["Diversity L2"], errors="coerce")
    df["Diversity L1"] = pd.to_numeric(df["Diversity L1"], errors="coerce")

    # Define the metrics to plot
    metrics = [
        ("Diversity L1", "L1 Norm"),
        ("Diversity L2", "L2 Norm"),
        ("MSE", "Mean Squared Error"),
        ("Convergence Rate", "Convergence Rate"),
    ]

    # Create scatter plots for each metric
    for metric_col, metric_name in metrics:
        plt.figure(figsize=(8, 6))
        plt.scatter(df["Label"], df[metric_col], alpha=0.7, edgecolors="k")
        plt.title(f"{metric_name} vs Label", fontsize=14)
        plt.xlabel("Label", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        output_path = f"{output_dir}/{metric_col.replace(' ', '_')}_vs_Label.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the evaluation results CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the plots.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    plot_evaluation_metrics(args.csv_path, args.output_dir)
