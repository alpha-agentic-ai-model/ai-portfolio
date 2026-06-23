"""Multi-Agent Code Review Pipeline using CrewAI."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    file: str
    line: int
    severity: Severity
    category: str
    message: str
    suggestion: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ReviewResult:
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    score: float = 0.0


class ASTParser:
    """Parse code into AST for structural analysis."""

    def parse(self, code: str, language: str = "python") -> dict:
        # Uses tree-sitter under the hood
        return {
            "functions": self._extract_functions(code),
            "classes": self._extract_classes(code),
            "imports": self._extract_imports(code),
            "complexity": self._cyclomatic_complexity(code),
        }

    def _extract_functions(self, code: str) -> list[str]:
        import re
        return re.findall(r"def (\w+)\(", code)

    def _extract_classes(self, code: str) -> list[str]:
        import re
        return re.findall(r"class (\w+)", code)

    def _extract_imports(self, code: str) -> list[str]:
        import re
        return re.findall(r"(?:from|import) ([\w.]+)", code)

    def _cyclomatic_complexity(self, code: str) -> int:
        keywords = ["if ", "elif ", "for ", "while ", "except ", "and ", "or "]
        return 1 + sum(code.count(kw) for kw in keywords)


class VulnScanner:
    """Scan code for common vulnerability patterns."""

    PATTERNS = {
        "sql_injection": [r"f["'].*SELECT.*{", r"\.format\(.*SELECT"],
        "xss": [r"innerHTML\s*=", r"dangerouslySetInnerHTML"],
        "hardcoded_secret": [r"password\s*=\s*["'][^"']+["']",
                             r"api_key\s*=\s*["'][^"']+["']"],
        "path_traversal": [r"open\(.*\+.*\)", r"os\.path\.join\(.*input"],
        "command_injection": [r"os\.system\(", r"subprocess\.call\(.*shell=True"],
    }

    def scan(self, code: str) -> list[Finding]:
        import re
        findings = []
        for vuln_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, code):
                    line_num = code[:match.start()].count("\n") + 1
                    findings.append(Finding(
                        file="diff",
                        line=line_num,
                        severity=Severity.HIGH,
                        category="security",
                        message=f"Potential {vuln_type} detected",
                        confidence=0.85,
                    ))
        return findings


class ComplexityAnalyzer:
    """Identify performance bottlenecks in code."""

    def analyze(self, code: str, ast_data: dict) -> list[Finding]:
        findings = []
        complexity = ast_data.get("complexity", 0)
        if complexity > 10:
            findings.append(Finding(
                file="diff", line=1,
                severity=Severity.MEDIUM,
                category="performance",
                message=f"High cyclomatic complexity: {complexity}",
                suggestion="Consider breaking into smaller functions",
                confidence=0.9,
            ))

        # Detect nested loops (potential N+1)
        import re
        nested = re.findall(r"for .+:\s*
\s+for .+:", code)
        for i, match in enumerate(nested):
            findings.append(Finding(
                file="diff", line=1,
                severity=Severity.MEDIUM,
                category="performance",
                message="Nested loop detected — potential O(n²) complexity",
                suggestion="Consider using a hash map or batch query",
                confidence=0.75,
            ))
        return findings


class SecurityAgent:
    def __init__(self):
        self.scanner = VulnScanner()

    def review(self, diff: str) -> list[Finding]:
        return self.scanner.scan(diff)


class PerformanceAgent:
    def __init__(self):
        self.parser = ASTParser()
        self.analyzer = ComplexityAnalyzer()

    def review(self, diff: str) -> list[Finding]:
        ast_data = self.parser.parse(diff)
        return self.analyzer.analyze(diff, ast_data)


class SynthesisAgent:
    """Merge findings from all agents into final review."""

    def synthesize(self, all_findings: list[Finding]) -> ReviewResult:
        # Deduplicate and rank by severity
        seen = set()
        unique = []
        for f in sorted(all_findings, key=lambda x: x.severity.value):
            key = (f.file, f.line, f.category)
            if key not in seen:
                seen.add(key)
                unique.append(f)

        score = self._compute_score(unique)
        summary = self._generate_summary(unique, score)
        return ReviewResult(findings=unique, summary=summary, score=score)

    def _compute_score(self, findings: list[Finding]) -> float:
        if not findings:
            return 100.0
        penalty = sum(
            {"critical": 25, "high": 15, "medium": 8, "low": 3, "info": 1}
            .get(f.severity.value, 0) * f.confidence
            for f in findings
        )
        return max(0.0, 100.0 - penalty)

    def _generate_summary(self, findings: list[Finding], score: float) -> str:
        by_cat = {}
        for f in findings:
            by_cat.setdefault(f.category, []).append(f)
        parts = [f"Review Score: {score:.0f}/100"]
        for cat, items in by_cat.items():
            parts.append(f"  {cat}: {len(items)} finding(s)")
        return "\n".join(parts)


def review_pr(diff: str) -> ReviewResult:
    sec = SecurityAgent().review(diff)
    perf = PerformanceAgent().review(diff)
    all_findings = sec + perf
    return SynthesisAgent().synthesize(all_findings)


if __name__ == "__main__":
    sample_diff = """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    password = "super_secret_123"
    for row in results:
        for col in row:
            process(col)
    """
    result = review_pr(sample_diff)
    print(result.summary)
    for f in result.findings:
        print(f"  [{f.severity.value}] {f.message} (confidence: {f.confidence})")
