"""
MODEL_001 授信额度测算 — 接口测试脚本
接口：POST http://127.0.0.1:5000/api/model/score
依赖：pip install requests
运行：python test_runner.py
日志：同目录下 test_runner.log
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import logging
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

# ──────────────────────────── 日志配置 ────────────────────────────

LOG_FILE = "test_runner.log"

logger = logging.getLogger("test_runner")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 控制台 handler
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)

# 文件 handler（记录 DEBUG 及以上）
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)

logger.addHandler(_ch)
logger.addHandler(_fh)

# ──────────────────────────── 常量 ────────────────────────────────

BASE_URL  = "http://127.0.0.1:5000"
SCORE_URL = f"{BASE_URL}/api/model/score"

# ──────────────────────────── 数据结构 ────────────────────────────

@dataclass
class Assertion:
    xpath: str
    expected: str
    desc: str


@dataclass
class TestCase:
    id: str
    name: str
    category: str
    db_setup: Optional[dict]
    request_xml: str
    assertions: list[Assertion]


@dataclass
class AssertResult:
    desc: str
    xpath: str
    expected: str
    actual: str
    passed: bool


@dataclass
class CaseResult:
    case: TestCase
    passed: bool
    assert_results: list[AssertResult] = field(default_factory=list)
    error: str = ""


# ──────────────────────────── 执行引擎 ────────────────────────────

def xml_get(root: ET.Element, xpath: str) -> str:
    tag = xpath.lstrip("/")
    el  = root.find(tag)
    return el.text.strip() if el is not None and el.text else ""


def run_case(tc: TestCase) -> CaseResult:
    sep = "-" * 60
    logger.info(sep)
    logger.info(f"[用例开始] {tc.id} | {tc.category} | {tc.name}")

    # ── 组装 payload ──
    payload: dict = {"request": tc.request_xml}
    if tc.db_setup:
        payload["db_setup"] = tc.db_setup
        logger.info(f"[db_setup] 写入前置数据:\n{json.dumps(tc.db_setup, ensure_ascii=False, indent=2)}")
    else:
        logger.info("[db_setup] 无前置数据，使用 DB 现有记录")

    logger.info(f"[payload] 完整请求体:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")

    # ── 发送请求 ──
    logger.info(f"[请求] POST {SCORE_URL}")
    logger.info(f"[请求头] Content-Type: application/json")

    try:
        resp = requests.post(SCORE_URL, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"[网络异常] {e}")
        return CaseResult(case=tc, passed=False, error=str(e))

    logger.info(f"[响应] HTTP {resp.status_code}  耗时 {resp.elapsed.total_seconds()*1000:.1f}ms")
    logger.info(f"[响应体]\n{resp.text}")

    if not resp.ok:
        err = f"HTTP {resp.status_code}"
        logger.error(f"[响应异常] {err}")
        return CaseResult(case=tc, passed=False, error=err)

    # ── 解析响应 XML ──
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as e:
        err = f"响应XML解析失败: {e}"
        logger.error(f"[解析失败] {err}")
        return CaseResult(case=tc, passed=False, error=err)

    # ── 断言 ──
    logger.info("[断言开始]")
    assert_results = []
    all_passed = True
    for a in tc.assertions:
        actual = xml_get(root, a.xpath)
        ok     = actual == a.expected
        if not ok:
            all_passed = False
        status = "PASS" if ok else "FAIL"
        logger.info(
            f"  [{status}] {a.desc} | xpath={a.xpath} "
            f"| 期望='{a.expected}' 实际='{actual}'"
        )
        assert_results.append(AssertResult(
            desc=a.desc, xpath=a.xpath,
            expected=a.expected, actual=actual, passed=ok,
        ))

    result_str = "PASS" if all_passed else "FAIL"
    logger.info(f"[用例结束] {tc.id} => {result_str}")
    return CaseResult(case=tc, passed=all_passed, assert_results=assert_results)


# ──────────────────────────── 测试用例定义 ────────────────────────────

def build_xml(user_id: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<REQUEST>'
        f'<user_id>{user_id}</user_id>'
        '<model_id>MODEL_001</model_id>'
        '<channel>131</channel>'
        '<business_params>'
        '<sx_trace_id></sx_trace_id>'
        '<cp_code>8890</cp_code>'
        '<c_code>110110</c_code>'
        '<p_code>110000</p_code>'
        '</business_params>'
        '</REQUEST>'
    )


TEST_CASES: list[TestCase] = [

    # ── 正常流程 ──────────────────────────────────────────────────

    TestCase(
        id="TC_MODEL001_001",
        name="工资<=10000且无社保不准入",
        category="正常流程",
        db_setup={
            "user_info":           {"user_id": "TC001", "user_level": 0},
            "account_balance":     {"user_id": "TC001", "avg_3m_balance": 0},
            "cgs_social_security": {"user_id": "TC001", "social_security_flag": 0},
            "salary_summary":      {"user_id": "TC001", "monthly_salary": 8000},
        },
        request_xml=build_xml("TC001"),
        assertions=[
            Assertion("//result_code", "0000", "接口调用成功"),
            Assertion("//admit_flag",  "0",    "准入判断：不准入"),
            Assertion("//credit_limit","0.00", "授信额度为零"),
        ],
    ),

    TestCase(
        id="TC_MODEL001_002",
        name="工资<=10000但有社保不准入",
        category="正常流程",
        db_setup={
            "user_info":           {"user_id": "TC002", "user_level": 0},
            "account_balance":     {"user_id": "TC002", "avg_3m_balance": 0},
            "cgs_social_security": {"user_id": "TC002", "social_security_flag": 1},
            "salary_summary":      {"user_id": "TC002", "monthly_salary": 9000},
        },
        request_xml=build_xml("TC002"),
        assertions=[
            Assertion("//result_code", "0000", "接口调用成功"),
            Assertion("//admit_flag",  "0",    "准入判断：不准入"),
            Assertion("//credit_limit","0.00", "授信额度为零"),
        ],
    ),

    TestCase(
        id="TC_MODEL001_003",
        name="工资>10000但无社保不准入",
        category="正常流程",
        db_setup={
            "user_info":           {"user_id": "TC003", "user_level": 0},
            "account_balance":     {"user_id": "TC003", "avg_3m_balance": 0},
            "cgs_social_security": {"user_id": "TC003", "social_security_flag": 0},
            "salary_summary":      {"user_id": "TC003", "monthly_salary": 12000},
        },
        request_xml=build_xml("TC003"),
        assertions=[
            Assertion("//result_code", "0000", "接口调用成功"),
            Assertion("//admit_flag",  "0",    "准入判断：不准入"),
            Assertion("//credit_limit","0.00", "授信额度为零"),
        ],
    ),

    TestCase(
        id="TC_MODEL001_004",
        name="工资>10000有社保资产>0准入",
        category="正常流程",
        db_setup={
            "user_info":           {"user_id": "TC004", "user_level": 3},
            "account_balance":     {"user_id": "TC004", "avg_3m_balance": 5000},
            "cgs_social_security": {"user_id": "TC004", "social_security_flag": 1},
            "salary_summary":      {"user_id": "TC004", "monthly_salary": 15000},
        },
        request_xml=build_xml("TC004"),
        assertions=[
            Assertion("//result_code", "0000",     "接口调用成功"),
            Assertion("//admit_flag",  "1",         "准入判断：准入"),
            Assertion("//credit_limit","517500.00", "授信额度：3x5000x15000x0.0023"),
        ],
    ),

    # ── 边界值 ────────────────────────────────────────────────────

    TestCase(
        id="BV_001",
        name="月薪恰好=10000（临界不准入）",
        category="边界值",
        db_setup={
            "user_info":           {"user_id": "BV001", "user_level": 2},
            "account_balance":     {"user_id": "BV001", "avg_3m_balance": 5000},
            "cgs_social_security": {"user_id": "BV001", "social_security_flag": 1},
            "salary_summary":      {"user_id": "BV001", "monthly_salary": 10000},
        },
        request_xml=build_xml("BV001"),
        assertions=[
            Assertion("//result_code", "0000", "接口调用成功"),
            Assertion("//admit_flag",  "0",    "月薪=10000 严格不满足>10000，不准入"),
            Assertion("//credit_limit","0.00", "不准入额度为零"),
        ],
    ),

    TestCase(
        id="BV_002",
        name="月薪=10001（临界+1，有社保，准入）",
        category="边界值",
        db_setup={
            "user_info":           {"user_id": "BV002", "user_level": 2},
            "account_balance":     {"user_id": "BV002", "avg_3m_balance": 5000},
            "cgs_social_security": {"user_id": "BV002", "social_security_flag": 1},
            "salary_summary":      {"user_id": "BV002", "monthly_salary": 10001},
        },
        request_xml=build_xml("BV002"),
        assertions=[
            Assertion("//result_code", "0000",     "接口调用成功"),
            Assertion("//admit_flag",  "1",         "月薪=10001 满足>10000，准入"),
            Assertion("//credit_limit","230023.00", "授信额度：2x5000x10001x0.0023"),
        ],
    ),

    TestCase(
        id="BV_003",
        name="准入但平均余额为负（授信额归零）",
        category="边界值",
        db_setup={
            "user_info":           {"user_id": "BV003", "user_level": 3},
            "account_balance":     {"user_id": "BV003", "avg_3m_balance": -1000},
            "cgs_social_security": {"user_id": "BV003", "social_security_flag": 1},
            "salary_summary":      {"user_id": "BV003", "monthly_salary": 20000},
        },
        request_xml=build_xml("BV003"),
        assertions=[
            Assertion("//result_code", "0000", "接口调用成功"),
            Assertion("//admit_flag",  "1",    "满足准入条件"),
            Assertion("//credit_limit","0.00", "计算结果为负，归零"),
        ],
    ),

    TestCase(
        id="BV_004",
        name="user_level=0 准入但额度为零",
        category="边界值",
        db_setup={
            "user_info":           {"user_id": "BV004", "user_level": 0},
            "account_balance":     {"user_id": "BV004", "avg_3m_balance": 10000},
            "cgs_social_security": {"user_id": "BV004", "social_security_flag": 1},
            "salary_summary":      {"user_id": "BV004", "monthly_salary": 20000},
        },
        request_xml=build_xml("BV004"),
        assertions=[
            Assertion("//result_code", "0000", "接口调用成功"),
            Assertion("//admit_flag",  "1",    "满足准入条件"),
            Assertion("//credit_limit","0.00", "user_level=0 乘积为零"),
        ],
    ),

    # ── 异常流程 ──────────────────────────────────────────────────

    TestCase(
        id="EX_001",
        name="user_id不存在（DB无记录）",
        category="异常流程",
        db_setup=None,
        request_xml=(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<REQUEST><user_id>NOT_EXIST</user_id><model_id>MODEL_001</model_id>'
            '<channel>131</channel><business_params></business_params></REQUEST>'
        ),
        assertions=[
            Assertion("//result_code", "9003", "返回用户数据不存在错误码"),
        ],
    ),

    TestCase(
        id="EX_002",
        name="user_id为空",
        category="异常流程",
        db_setup=None,
        request_xml=(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<REQUEST><user_id></user_id><model_id>MODEL_001</model_id>'
            '<channel>131</channel><business_params></business_params></REQUEST>'
        ),
        assertions=[
            Assertion("//result_code", "9001", "返回user_id为空错误码"),
        ],
    ),

    TestCase(
        id="EX_003",
        name="不支持的model_id",
        category="异常流程",
        db_setup=None,
        request_xml=(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<REQUEST><user_id>TC001</user_id><model_id>MODEL_999</model_id>'
            '<channel>131</channel><business_params></business_params></REQUEST>'
        ),
        assertions=[
            Assertion("//result_code", "9002", "返回不支持model_id错误码"),
        ],
    ),

    TestCase(
        id="EX_004",
        name="request字段为非法XML",
        category="异常流程",
        db_setup=None,
        request_xml="this is not xml at all",
        assertions=[
            Assertion("//result_code", "9999", "返回XML解析失败错误码"),
        ],
    ),

    TestCase(
        id="EX_005",
        name="缺少model_id节点（默认空字符串不匹配）",
        category="异常流程",
        db_setup=None,
        request_xml=(
            '<?xml version="1.0" encoding="utf-8"?>'
            '<REQUEST><user_id>TC001</user_id>'
            '<channel>131</channel><business_params></business_params></REQUEST>'
        ),
        assertions=[
            Assertion("//result_code", "9002", "model_id缺失视为不支持"),
        ],
    ),
]


# ──────────────────────────── 报告输出 ────────────────────────────

SEP = "-" * 72


def print_report(results: list[CaseResult]):
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    logger.info("=" * 72)
    logger.info(f"  测试报告  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  接口：{SCORE_URL}")
    logger.info(f"  总计: {total}  通过: {passed}  失败: {failed}")
    logger.info("=" * 72)

    categories: dict[str, list[CaseResult]] = {}
    for r in results:
        categories.setdefault(r.case.category, []).append(r)

    for cat, cat_results in categories.items():
        logger.info(f"\n【{cat}】")
        logger.info(SEP)
        for r in cat_results:
            icon = "[OK]" if r.passed else "[NG]"
            logger.info(f"  {icon} {r.case.id}  {r.case.name}")
            if r.error:
                logger.error(f"       错误：{r.error}")
            for ar in r.assert_results:
                ar_icon = "  [pass]" if ar.passed else "  [fail]"
                if ar.passed:
                    logger.info(f"       {ar_icon} {ar.desc}  ({ar.xpath} = '{ar.actual}')")
                else:
                    logger.warning(f"       {ar_icon} {ar.desc}")
                    logger.warning(f"            期望: '{ar.expected}'")
                    logger.warning(f"            实际: '{ar.actual}'")

    conclusion = "全部通过" if failed == 0 else f"{failed} 条失败，请检查上方详情"
    logger.info("=" * 72)
    logger.info(f"  结论: {conclusion}")
    logger.info("=" * 72)


# ──────────────────────────── 主入口 ────────────────────────────

if __name__ == "__main__":
    logger.info(f"{'=' * 72}")
    logger.info(f"  测试启动  目标服务: {SCORE_URL}")
    logger.info(f"  共 {len(TEST_CASES)} 条用例  日志文件: {LOG_FILE}")
    logger.info(f"{'=' * 72}")

    results = [run_case(tc) for tc in TEST_CASES]
    print_report(results)
