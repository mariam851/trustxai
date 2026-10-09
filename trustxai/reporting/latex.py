"""
trustxai.reporting.latex
~~~~~~~~~~~~~~~~~~~~~~~~
Automated LaTeX table generators for academic benchmarks and publications.
"""

from typing import Dict, List
import numpy as np
from trustxai.core.index import TrustEvaluationResult, TrustStatus


class AcademicReporter:
    """
    Generates publication-quality LaTeX tables summarizing Trust Benchmark metrics.
    """

    @staticmethod
    def generate_summary_table(
        results_by_model: Dict[str, List[TrustEvaluationResult]],
        caption: str = "Explanation Robustness and Reliability Benchmark Results",
        label: str = "tab:trustxai_benchmark",
    ) -> str:
        """
        Compiles model audit outputs into a booktabs LaTeX table.

        Parameters
        ----------
        results_by_model : Dict[str, List[TrustEvaluationResult]]
            Dictionary mapping model names to their audit result lists.
        """
        lines = [
            "\\begin{table*}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Mean UTI} $\\uparrow$ & \\textbf{Certified (\\%)} $\\uparrow$ & \\textbf{Fragile (\\%)} $\\downarrow$ & \\textbf{Ambiguous/OOD (\\%)} $\\downarrow$ \\\\",
            "\\midrule",
        ]

        for model_name, results in results_by_model.items():
            total = len(results)
            uti_mean = float(np.mean([r.uti_score for r in results]))

            n_certified = sum(1 for r in results if r.status == TrustStatus.CERTIFIED)
            n_fragile = sum(1 for r in results if r.status == TrustStatus.FRAGILE)
            n_ambiguous = sum(1 for r in results if r.status == TrustStatus.AMBIGUOUS_OR_OOD)

            pct_cert = (n_certified / total) * 100.0
            pct_frag = (n_fragile / total) * 100.0
            pct_amb = (n_ambiguous / total) * 100.0

            row = (
                f"{model_name} & "
                f"{uti_mean:.3f} & "
                f"{pct_cert:.1f}\\% & "
                f"{pct_frag:.1f}\\% & "
                f"{pct_amb:.1f}\\% \\\\"
            )
            lines.append(row)

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ])

        return "\n".join(lines)
    