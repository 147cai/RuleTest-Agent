"""
YAML 驱动的接口测试执行器

支持两种调用方式：
  1. 命令行：python yaml_runner.py --file path/to/test_case.yaml [--server http://127.0.0.1:5000]
  2. 模块调用：from yaml_runner import run_yaml_file, run_yaml_content
"""

import sys
import json
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests
import yaml


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class AssertResult:
    desc: str
    path: str
    op: str
    expected: str
    actual: str
    passed: bool


@dataclass
class CaseResult:
    case_id: str
    name: str
    category: str
    passed: bool
    assert_results: list = field(default_factory=list)
    error: str = ""
    status_code: int = 0
    request_payload: dict = field(default_factory=dict)
    response_body: str = ""


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _xml_get(root: ET.Element, xpath: str) -> str:
    """从 XML 根节点按 xpath 取文本，支持 //tag 和 /tag 格式。"""
    tag = xpath.lstrip("/")
    el = root.find(tag)
    return (el.text or "").strip() if el is not None else ""


def _assert(root: ET.Element, a: dict) -> AssertResult:
    """执行单条断言，目前支持 op=eq。"""
    path = a.get("path", "")
    op = a.get("op", "eq")
    expected = str(a.get("value", ""))
    desc = a.get("desc", path)
    actual = _xml_get(root, path)
    if op == "eq":
        passed = actual == expected
    elif op == "contains":
        passed = expected in actual
    elif op == "ne":
        passed = actual != expected
    else:
        passed = actual == expected  # 默认等值比较
    return AssertResult(desc=desc, path=path, op=op,
                        expected=expected, actual=actual, passed=passed)


def _build_db_setup(preconditions: dict) -> dict:
    """
    兼容两种格式：
      格式A（实际生成）：preconditions.user_info / account_balance / ...
      格式B（模板）：    preconditions.db_setup.user_info / ...
    """
    if not preconditions:
        return {}
    if "db_setup" in preconditions:
        return preconditions["db_setup"]
    # 格式A：4个表的key直接在preconditions下
    keys = {"user_info", "account_balance", "cgs_social_security", "salary_summary"}
    result = {k: preconditions[k] for k in keys if k in preconditions}
    return result


# ── 核心执行 ──────────────────────────────────────────────────────────────────

def run_case(tc: dict, base_url: str, endpoint: str) -> CaseResult:
    case_id = str(tc.get("id", ""))
    name = tc.get("name", "")
    category = tc.get("category", "")

    preconditions = tc.get("preconditions") or {}
    db_setup = _build_db_setup(preconditions)

    request_section = tc.get("request") or {}
    request_xml = request_section.get("body", "")

    assertions = tc.get("assertions") or []

    # 组装 payload
    payload = {"request": request_xml}
    if db_setup:
        payload["db_setup"] = db_setup

    url = base_url.rstrip("/") + endpoint

    try:
        resp = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        return CaseResult(case_id=case_id, name=name, category=category,
                          passed=False, error=str(e), request_payload=payload)

    if not resp.ok:
        return CaseResult(case_id=case_id, name=name, category=category,
                          passed=False, error=f"HTTP {resp.status_code}",
                          status_code=resp.status_code,
                          request_payload=payload, response_body=resp.text)

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        return CaseResult(case_id=case_id, name=name, category=category,
                          passed=False, error=f"响应XML解析失败: {e}",
                          status_code=resp.status_code,
                          request_payload=payload, response_body=resp.text)

    assert_results = [_assert(root, a) for a in assertions]
    passed = all(r.passed for r in assert_results)

    return CaseResult(
        case_id=case_id, name=name, category=category,
        passed=passed, assert_results=assert_results,
        status_code=resp.status_code,
        request_payload=payload, response_body=resp.text,
    )


def _run_suite(suite_doc: dict, base_url: str) -> dict:
    """执行一个 YAML 文档（单个 suite）。"""
    suite = suite_doc.get("suite") or {}
    endpoint = suite.get("endpoint", "/api/model/score")
    test_cases = suite_doc.get("test_cases") or []

    results = [run_case(tc, base_url, endpoint) for tc in test_cases]

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)

    return {
        "suite_id": suite.get("id", ""),
        "suite_name": suite.get("name", ""),
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "results": [_case_result_to_dict(r) for r in results],
    }


def _case_result_to_dict(r: CaseResult) -> dict:
    return {
        "case_id": r.case_id,
        "name": r.name,
        "category": r.category,
        "passed": r.passed,
        "error": r.error,
        "status_code": r.status_code,
        "request_payload": r.request_payload,
        "response_body": r.response_body,
        "assert_results": [
            {
                "desc": a.desc,
                "path": a.path,
                "op": a.op,
                "expected": a.expected,
                "actual": a.actual,
                "passed": a.passed,
            }
            for a in r.assert_results
        ],
    }


# ── 对外接口 ──────────────────────────────────────────────────────────────────

def run_yaml_content(yaml_content: str, base_url: str = "http://127.0.0.1:5000") -> dict:
    """从 YAML 字符串执行测试，返回结果字典。"""
    try:
        doc = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        return {"error": f"YAML解析失败: {e}", "total": 0, "passed": 0, "failed": 0, "results": []}
    if not doc:
        return {"error": "YAML内容为空", "total": 0, "passed": 0, "failed": 0, "results": []}
    return _run_suite(doc, base_url)


def run_yaml_file(file_path: str, base_url: str = "http://127.0.0.1:5000") -> dict:
    """从 YAML 文件执行测试，返回结果字典。"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return {"error": f"读取文件失败: {e}", "total": 0, "passed": 0, "failed": 0, "results": []}
    return run_yaml_content(content, base_url)


# ── 命令行入口 ────────────────────────────────────────────────────────────────

def _print_report(report: dict):
    sep = "-" * 64
    print(sep)
    print(f"Suite: {report.get('suite_name', '')}  [{report.get('suite_id', '')}]")
    print(f"总计: {report['total']}  通过: {report['passed']}  失败: {report['failed']}")
    print(sep)
    for r in report.get("results", []):
        icon = "✓" if r["passed"] else "✗"
        print(f"  {icon} [{r['case_id']}] {r['name']}")
        if r["error"]:
            print(f"      错误: {r['error']}")
        for a in r.get("assert_results", []):
            a_icon = "  pass" if a["passed"] else "  FAIL"
            if a["passed"]:
                print(f"      {a_icon}  {a['desc']}  ({a['path']} = '{a['actual']}')")
            else:
                print(f"      {a_icon}  {a['desc']}")
                print(f"             期望: '{a['expected']}'  实际: '{a['actual']}'")
    print(sep)
    conclusion = "全部通过 ✓" if report["failed"] == 0 else f"{report['failed']} 条失败"
    print(f"结论: {conclusion}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YAML 驱动接口测试执行器")
    parser.add_argument("--file", required=True, help="YAML 测试文件路径")
    parser.add_argument("--server", default="http://127.0.0.1:5000", help="被测服务地址")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    args = parser.parse_args()

    report = run_yaml_file(args.file, args.server)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if "error" in report and report["total"] == 0:
            print(f"错误: {report['error']}", file=sys.stderr)
            sys.exit(1)
        _print_report(report)
        sys.exit(0 if report["failed"] == 0 else 1)
