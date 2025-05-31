import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
import transformers
from jax import numpy as jnp

import easydel as ed


def plot_results(results_df, output_dir="plots"):
    """
    Create comprehensive visualizations of benchmark results

    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """

    os.makedirs(output_dir, exist_ok=True)

    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    avg_results = (
        results_df.groupby(["prefill_length", "max_new_tokens"])
        .agg(
            {
                "tokens_per_second": ["mean", "std"],
                "total_time": ["mean", "std"],
                "tokens_generated": ["mean"],
            }
        )
        .reset_index()
    )

    # Flatten the multi-level columns
    avg_results.columns = ["_".join(col).strip("_") for col in avg_results.columns.values]

    # 1. TPS by prefill length for different max_new_tokens
    plt.figure(figsize=(12, 8))
    for max_tokens in sorted(results_df["max_new_tokens"].unique()):
        subset = avg_results[avg_results["max_new_tokens"] == max_tokens]
        plt.plot(
            subset["prefill_length"],
            subset["tokens_per_second_mean"],
            marker="o",
            linewidth=2,
            label=f"max_new_tokens={max_tokens}",
        )
        # Add error area using standard deviation
        plt.fill_between(
            subset["prefill_length"],
            subset["tokens_per_second_mean"] - subset["tokens_per_second_std"],
            subset["tokens_per_second_mean"] + subset["tokens_per_second_std"],
            alpha=0.2,
        )

    plt.xlabel("Prefill Length (tokens)")
    plt.ylabel("Tokens Per Second")
    plt.title("TPS vs Prefill Length for Different Max New Token Values")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Max New Tokens")

    # Use log scale for x-axis if prefill lengths span multiple orders of magnitude
    if max(results_df["prefill_length"]) / min(results_df["prefill_length"]) > 20:
        plt.xscale("log", base=2)
        plt.xticks(
            sorted(results_df["prefill_length"].unique()),
            sorted(results_df["prefill_length"].unique()),
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/tps_vs_prefill.png", dpi=300)
    plt.savefig(f"{output_dir}/tps_vs_prefill.pdf")

    # 2. TPS by max_new_tokens for different prefill lengths
    plt.figure(figsize=(12, 8))
    for prefill in sorted(results_df["prefill_length"].unique()):
        subset = avg_results[avg_results["prefill_length"] == prefill]
        plt.plot(
            subset["max_new_tokens"],
            subset["tokens_per_second_mean"],
            marker="o",
            linewidth=2,
            label=f"prefill={prefill}",
        )
        # Add error area
        plt.fill_between(
            subset["max_new_tokens"],
            subset["tokens_per_second_mean"] - subset["tokens_per_second_std"],
            subset["tokens_per_second_mean"] + subset["tokens_per_second_std"],
            alpha=0.2,
        )

    plt.xlabel("Max New Tokens")
    plt.ylabel("Tokens Per Second")
    plt.title("TPS vs Max New Tokens for Different Prefill Lengths")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Prefill Length")

    # Use log scale for x-axis if max_new_tokens span multiple orders of magnitude
    if max(results_df["max_new_tokens"]) / min(results_df["max_new_tokens"]) > 20:
        plt.xscale("log", base=2)
        plt.xticks(
            sorted(results_df["max_new_tokens"].unique()),
            sorted(results_df["max_new_tokens"].unique()),
        )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/tps_vs_max_new_tokens.png", dpi=300)
    plt.savefig(f"{output_dir}/tps_vs_max_new_tokens.pdf")

    # 3. High-quality heatmap of TPS
    plt.figure(figsize=(10, 8))
    pivot_table = avg_results.pivot(index="prefill_length", columns="max_new_tokens", values="tokens_per_second_mean")

    # Create heatmap with seaborn
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cmap="viridis",
        cbar_kws={"label": "Tokens Per Second"},
    )

    plt.title("TPS Heatmap by Prefill Length and Max New Tokens")
    plt.xlabel("Max New Tokens")
    plt.ylabel("Prefill Length")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tps_heatmap.png", dpi=300)
    plt.savefig(f"{output_dir}/tps_heatmap.pdf")

    # 4. Total generation time heatmap
    plt.figure(figsize=(10, 8))
    pivot_table_time = avg_results.pivot(index="prefill_length", columns="max_new_tokens", values="total_time_mean")

    ax = sns.heatmap(
        pivot_table_time,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cmap="rocket_r",
        cbar_kws={"label": "Generation Time (s)"},
    )

    plt.title("Generation Time Heatmap (seconds)")
    plt.xlabel("Max New Tokens")
    plt.ylabel("Prefill Length")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_heatmap.png", dpi=300)
    plt.savefig(f"{output_dir}/time_heatmap.pdf")

    # 5. 3D Surface plot for TPS
    try:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        X = sorted(results_df["prefill_length"].unique())
        Y = sorted(results_df["max_new_tokens"].unique())
        X, Y = np.meshgrid(X, Y)

        # Fill Z values from pivot table
        Z = np.zeros_like(X, dtype=float)
        for i, max_tokens in enumerate(sorted(results_df["max_new_tokens"].unique())):
            for j, prefill in enumerate(sorted(results_df["prefill_length"].unique())):
                try:
                    Z[i, j] = pivot_table.loc[prefill, max_tokens]
                except (KeyError, ValueError):
                    Z[i, j] = np.nan

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Tokens Per Second")

        # Add labels
        ax.set_xlabel("Prefill Length")
        ax.set_ylabel("Max New Tokens")
        ax.set_zlabel("Tokens Per Second")
        ax.set_title("3D Surface Plot of TPS Performance")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/tps_3d_surface.png", dpi=300)
        plt.savefig(f"{output_dir}/tps_3d_surface.pdf")
    except Exception as e:
        print(f"Skipping 3D plot due to error: {e}")

    plt.figure(figsize=(10, 8))
    avg_results["generation_time"] = avg_results["tokens_generated_mean"] / avg_results["tokens_per_second_mean"]
    avg_results["prefill_time"] = avg_results["total_time_mean"] - avg_results["generation_time"]
    avg_results["prefill_ratio"] = avg_results["prefill_time"] / avg_results["total_time_mean"] * 100

    pivot_prefill_ratio = avg_results.pivot(index="prefill_length", columns="max_new_tokens", values="prefill_ratio")

    ax = sns.heatmap(
        pivot_prefill_ratio,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cmap="coolwarm",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Prefill Time %"},
    )

    plt.title("Prefill Time as Percentage of Total Time")
    plt.xlabel("Max New Tokens")
    plt.ylabel("Prefill Length")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prefill_ratio_heatmap.png", dpi=300)
    plt.savefig(f"{output_dir}/prefill_ratio_heatmap.pdf")

    # 7. Boxplot of TPS variation across runs for each configuration
    plt.figure(figsize=(14, 10))

    # Prepare data for boxplot - combine prefill and max_new_tokens into a single label
    results_df["config"] = results_df["prefill_length"].astype(str) + "_" + results_df["max_new_tokens"].astype(str)

    config_avg_tps = results_df.groupby("config")["tokens_per_second"].mean().sort_values(ascending=False)
    top_configs = config_avg_tps.index[:15]
    plot_data = results_df[results_df["config"].isin(top_configs)]

    sns.boxplot(
        x="config",
        y="tokens_per_second",
        data=plot_data,
        palette="viridis",
        order=top_configs,
    )

    plt.xlabel("Configuration (prefill_maxnewtokens)")
    plt.ylabel("Tokens Per Second")
    plt.title("TPS Variation Across Runs for Top Configurations")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tps_variation_boxplot.png", dpi=300)
    plt.savefig(f"{output_dir}/tps_variation_boxplot.pdf")

    return avg_results


def run_benchmark(
    model_name,
    prefill_lengths,
    max_new_tokens_list,
    num_runs=3,
    sharding_axis_dims=(1, 1, 1, 1, -1),
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    temperature=0.7,
    top_p=0.95,
    top_k=10,
):
    """
    Run a benchmark for EasyDeL model with various prefill lengths and max_new_tokens.

    Args:
        model_name: HuggingFace model name
        prefill_lengths: List of prefill lengths to test
        max_new_tokens_list: List of max_new_tokens values to test
        num_runs: Number of runs for each configuration to average results
        sharding_axis_dims: Sharding configuration
        dtype: Data type for model operations
        param_dtype: Data type for model parameters
        temperature, top_p, top_k: Generation parameters

    Returns:
        DataFrame with benchmark results
    """
    # Set up model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = max(prefill_lengths) + max(max_new_tokens_list)

    print(f"Loading model: {model_name}")
    print(f"Max length set to: {max_length}")

    results = []

    # Create a sample message with adjustable length
    system_message = "You are a helpful AI assistant."
    base_prompt = "write 100 lines story about why you love EasyDeL"

    # Run benchmarks for each configuration
    for prefill_length in prefill_lengths:
        model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            model_name,
            auto_shard_model=True,
            sharding_axis_dims=sharding_axis_dims,
            config_kwargs=ed.EasyDeLBaseConfigDict(
                freq_max_position_embeddings=prefill_length + max(max_new_tokens_list),
                mask_max_position_embeddings=prefill_length + max(max_new_tokens_list),
                kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
                attn_dtype=dtype,
                attn_mechanism=ed.AttentionMechanisms.AUTO,
                blocksize_q=1024,
                blocksize_k=1024,
            ),
            quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            param_dtype=param_dtype,
            dtype=dtype,
        )
        for max_new_tokens in max_new_tokens_list:
            print(f"\nBenchmarking: prefill={prefill_length}, max_new_tokens={max_new_tokens}")

            inference = ed.vInference(
                model=model,
                processor_class=tokenizer,
                generation_config=ed.vInferenceConfig(
                    max_new_tokens=max_new_tokens,
                    sampling_params=ed.SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=0.0,
                        top_p=1,
                        top_k=0,
                    ),  # GREADY
                    eos_token_id=model.generation_config.eos_token_id,
                    streaming_chunks=64,
                    num_return_sequences=1,
                ),
                inference_name=f"bench{prefill_length}{max_new_tokens}",
            )

            # Apply chat template to get tokens
            ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": base_prompt},
                ],
                return_tensors="jax",
                return_dict=True,
                add_generation_prompt=True,
            )

            # Get actual prefill length
            actual_prefill_length = ids.input_ids.shape[1]
            print(f"Actual prefill length: {actual_prefill_length}")

            # Precompile for this specific length
            inference.precompile(
                ed.vInferencePreCompileConfig(
                    batch_size=1,
                    prefill_length=prefill_length,
                )
            )

            # Run multiple times and average results
            run_results = []
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}...")

                start_time = time.time()
                # The generate() call is expected to yield a response object
                for response in inference.generate(**ids):  # noqa
                    pass
                end_time = time.time()

                # Store results
                run_results.append(
                    {
                        "prefill_length": int(prefill_length),
                        "max_new_tokens": int(max_new_tokens),
                        "total_time": float(end_time - start_time),
                        "tokens_generated": int(response.generated_tokens),
                        "tokens_per_second": float(response.tokens_per_second),
                        "run": int(run + 1),
                    }
                )

                print(f"  TPS: {response.tokens_per_second}, Time: {end_time - start_time}s")
            del inference
            # Add all runs to results
            results.extend(run_results)
        del model
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="EasyDeL Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M", help="Model name or path")
    parser.add_argument(
        "--prefill_lengths",
        nargs="+",
        type=int,
        default=[2**s for s in range(13, 17)],
        help="List of prefill lengths to benchmark",
    )
    parser.add_argument(
        "--max_new_tokens_list",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192],
        help="List of max_new_tokens values to benchmark",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="Output file for results",
    )
    parser.add_argument(
        "--plot_dir",
        default="plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    print("Starting benchmark with:")
    print(f"  Model: {args.model}")
    print(f"  Prefill lengths: {args.prefill_lengths}")
    print(f"  Max new tokens: {args.max_new_tokens_list}")
    print(f"  Runs per config: {args.num_runs}")

    # Run the benchmark
    results_df = run_benchmark(
        model_name=args.model,
        prefill_lengths=args.prefill_lengths,
        max_new_tokens_list=args.max_new_tokens_list,
        num_runs=args.num_runs,
    )

    # Save raw results
    results_df.to_csv(args.output, index=False)
    plot_results(results_df, "plots")


if __name__ == "__main__":
    main()
