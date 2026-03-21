import pytest
import datetime
import os

RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'RESULTS.md')


class ResultsCollector:
    def __init__(self):
        self.results = []

    def add(self, name, passed, duration):
        self.results.append((name, passed, duration))


collector = ResultsCollector()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == 'call':
        collector.add(report.nodeid, report.passed, report.duration)


def pytest_sessionfinish(session, exitstatus):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_pass = sum(1 for _, p, _ in collector.results if p)
    n_fail = sum(1 for _, p, _ in collector.results if not p)
    total_time = sum(d for _, _, d in collector.results)

    lines = [
        f"\n---\n",
        f"### Unit Tests (pytest)",
        f"*Run: {timestamp}*\n",
        f"- **Total**: {len(collector.results)} tests",
        f"- **Passed**: {n_pass} ✅",
        f"- **Failed**: {n_fail} {'❌' if n_fail > 0 else ''}",
        f"- **Duration**: {total_time:.2f}s\n",
    ]

    for name, passed, duration in collector.results:
        icon = "✅" if passed else "❌"
        short_name = name.split("::")[-1]
        lines.append(f"- {icon} `{short_name}` ({duration:.3f}s)")

    lines.append("")

    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
