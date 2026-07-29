import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import asyncio


@dataclass
class AnalysisStep:
    name: str
    action: str
    params: dict
    priority: int = 1


@dataclass
class EDAReport:
    summary: dict[str, Any]
    outliers: list[dict]
    correlations: pd.DataFrame
    charts: list[str]
    narrative: str


class DataAnalystAgent:
    """Autonomous EDA agent with plan-execute loop and tool calling."""

    def __init__(self, llm_client, output_dir: str = "./reports"):
        self.llm = llm_client
        self.output_dir = output_dir
        self.tools: dict[str, Callable] = {
            "describe": self.describe,
            "correlate": self.correlate,
            "detect_outliers": self.detect_outliers,
            "plot_distribution": self.plot_distribution,
            "plot_correlation_matrix": self.plot_correlation_matrix,
            "test_normality": self.test_normality,
            "cardinality_analysis": self.cardinality_analysis,
        }

    async def analyze(self, df: pd.DataFrame) -> EDAReport:
        """Run full automated EDA pipeline."""
        plan = await self.llm.plan_analysis(
            schema=df.dtypes.to_dict(),
            sample=df.head(3).to_dict(),
            shape=df.shape,
        )

        results = {}
        charts = []

        for step in plan.steps:
            tool = self.tools.get(step.action)
            if tool is None:
                continue
            result = await tool(df, **step.params)
            results[step.name] = result
            if isinstance(result, str) and result.endswith(".png"):
                charts.append(result)

        narrative = await self.llm.narrate(results)
        return EDAReport(
            summary=results.get("describe", {}),
            outliers=results.get("detect_outliers", []),
            correlations=results.get("correlate", pd.DataFrame()),
            charts=charts,
            narrative=narrative,
        )

    async def describe(self, df: pd.DataFrame, **kwargs) -> dict:
        """Generate statistical summary for all columns."""
        numeric_summary = df.describe().to_dict()
        categorical_summary = {}
        for col in df.select_dtypes(include=["object", "category"]).columns:
            categorical_summary[col] = {
                "unique": df[col].nunique(),
                "top": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "null_pct": round(df[col].isnull().mean() * 100, 2),
            }
        return {"numeric": numeric_summary, "categorical": categorical_summary}

    async def correlate(self, df: pd.DataFrame, method: str = "pearson", **kwargs) -> pd.DataFrame:
        """Compute correlation matrix for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)

    async def detect_outliers(self, df: pd.DataFrame, columns: list[str] = None, method: str = "iqr", **kwargs) -> list[dict]:
        """Detect outliers using IQR or Z-score method."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = []
        for col in columns:
            series = df[col].dropna()
            if method == "iqr":
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                mask = (series < lower) | (series > upper)
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(series))
                mask = z_scores > 3
            else:
                continue

            outlier_rows = df.loc[mask, [col]]
            for idx, row in outlier_rows.iterrows():
                outliers.append({"index": idx, "column": col, "value": row[col]})

        return outliers

    async def plot_distribution(self, df: pd.DataFrame, columns: list[str] = None, **kwargs) -> str:
        """Generate distribution plots for specified columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:6]

        n_cols = min(len(columns), 3)
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)

        for i, col in enumerate(columns):
            ax = axes[i // n_cols, i % n_cols]
            df[col].dropna().hist(bins=30, ax=ax, color="#6c63ff", alpha=0.7, edgecolor="white")
            ax.set_title(col, fontsize=10)
            ax.tick_params(labelsize=8)

        plt.tight_layout()
        path = f"{self.output_dir}/distributions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    async def plot_correlation_matrix(self, df: pd.DataFrame, **kwargs) -> str:
        """Generate a heatmap of the correlation matrix."""
        corr = await self.correlate(df)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(im)
        plt.tight_layout()
        path = f"{self.output_dir}/correlation_matrix.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    async def test_normality(self, df: pd.DataFrame, columns: list[str] = None, **kwargs) -> dict:
        """Run Shapiro-Wilk normality tests on numeric columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        results = {}
        for col in columns:
            sample = df[col].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000, random_state=42)
            if len(sample) < 8:
                continue
            stat, p_value = stats.shapiro(sample)
            results[col] = {
                "statistic": round(stat, 4),
                "p_value": round(p_value, 6),
                "is_normal": p_value > 0.05,
            }
        return results

    async def cardinality_analysis(self, df: pd.DataFrame, **kwargs) -> dict:
        """Analyze cardinality of all columns for feature engineering hints."""
        analysis = {}
        for col in df.columns:
            n_unique = df[col].nunique()
            analysis[col] = {
                "unique_values": n_unique,
                "cardinality_ratio": round(n_unique / len(df), 4),
                "null_count": int(df[col].isnull().sum()),
                "dtype": str(df[col].dtype),
                "suggested_encoding": (
                    "drop" if n_unique == 1
                    else "binary" if n_unique == 2
                    else "one_hot" if n_unique <= 10
                    else "target" if n_unique <= 50
                    else "hash" if df[col].dtype == "object"
                    else "numeric"
                ),
            }
        return analysis
