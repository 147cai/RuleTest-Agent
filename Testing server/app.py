"""
被测系统 Demo — MODEL_001 授信额度测算

单一接口：POST /api/model/score  Content-Type: application/json

请求体：
{
  "db_setup": {                        <- 可选，存在则写入/覆盖前置数据
    "user_info":           {...},
    "account_balance":     {...},
    "cgs_social_security": {...},
    "salary_summary":      {...}
  },
  "request": "<xml>...</xml>"          <- 必填，业务请求 XML 字符串
}

响应：application/xml
日志：同目录下 app.log
"""

import json
import logging
import sqlite3
import xml.etree.ElementTree as ET
from flask import Flask, request, Response, jsonify

# ──────────────────────────── 日志配置 ────────────────────────────

LOG_FILE = "app.log"

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)

logger.addHandler(_ch)
logger.addHandler(_fh)

# 关闭 Flask/Werkzeug 默认日志，避免与自定义日志重复
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# ──────────────────────────── 应用初始化 ────────────────────────────

app     = Flask(__name__)
DB_PATH = "test_demo.db"


# ──────────────────────────── DB ────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_info (
            user_id    TEXT PRIMARY KEY,
            user_level INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS account_balance (
            user_id        TEXT PRIMARY KEY,
            avg_3m_balance REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cgs_social_security (
            user_id              TEXT PRIMARY KEY,
            social_security_flag INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS salary_summary (
            user_id        TEXT PRIMARY KEY,
            monthly_salary REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    logger.info("[DB] 数据库初始化完成，路径: %s", DB_PATH)


def write_db_setup(conn, setup: dict):
    if "user_info" in setup:
        d = setup["user_info"]
        conn.execute("INSERT OR REPLACE INTO user_info VALUES (?,?)",
                     (d["user_id"], d["user_level"]))
        logger.debug("[DB] user_info        <- user_id=%s  user_level=%s",
                     d["user_id"], d["user_level"])

    if "account_balance" in setup:
        d = setup["account_balance"]
        conn.execute("INSERT OR REPLACE INTO account_balance VALUES (?,?)",
                     (d["user_id"], d["avg_3m_balance"]))
        logger.debug("[DB] account_balance  <- user_id=%s  avg_3m_balance=%s",
                     d["user_id"], d["avg_3m_balance"])

    if "cgs_social_security" in setup:
        d = setup["cgs_social_security"]
        conn.execute("INSERT OR REPLACE INTO cgs_social_security VALUES (?,?)",
                     (d["user_id"], d["social_security_flag"]))
        logger.debug("[DB] cgs_social_security <- user_id=%s  social_security_flag=%s",
                     d["user_id"], d["social_security_flag"])

    if "salary_summary" in setup:
        d = setup["salary_summary"]
        conn.execute("INSERT OR REPLACE INTO salary_summary VALUES (?,?)",
                     (d["user_id"], d["monthly_salary"]))
        logger.debug("[DB] salary_summary   <- user_id=%s  monthly_salary=%s",
                     d["user_id"], d["monthly_salary"])


# ──────────────────────────── 工具函数 ────────────────────────────

def xml_resp(result_code, admit_flag=None, credit_limit=None, error_msg=None):
    root = ET.Element("RESPONSE")
    ET.SubElement(root, "result_code").text = result_code
    if error_msg is not None:
        ET.SubElement(root, "error_msg").text = error_msg
    if admit_flag is not None:
        ET.SubElement(root, "admit_flag").text = str(admit_flag)
    if credit_limit is not None:
        ET.SubElement(root, "credit_limit").text = f"{credit_limit:.2f}"
    body = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")

    logger.info("[响应] result_code=%s%s%s%s",
                result_code,
                f"  admit_flag={admit_flag}"       if admit_flag   is not None else "",
                f"  credit_limit={credit_limit:.2f}" if credit_limit is not None else "",
                f"  error_msg={error_msg}"          if error_msg    is not None else "")
    logger.debug("[响应体]\n%s", body)

    return Response(body, content_type="application/xml; charset=utf-8")


# ──────────────────────────── 单一接口 ────────────────────────────

@app.route("/api/model/score", methods=["POST"])
def model_score():
    sep = "-" * 60
    logger.info(sep)
    logger.info("[请求] POST /api/model/score  from %s", request.remote_addr)
    logger.info("[请求头] Content-Type: %s", request.content_type)

    # ── 解析外层 JSON ──
    payload = request.get_json(silent=True)
    if not payload or "request" not in payload:
        logger.warning("[校验] 请求体非法：缺少 request 字段，原始内容: %s",
                       request.data.decode("utf-8", errors="replace")[:200])
        return xml_resp("9999", error_msg="请求体须为JSON且包含 request 字段")

    logger.debug("[请求体]\n%s", json.dumps(payload, ensure_ascii=False, indent=2))

    # ── 写入前置数据（如有）──
    if "db_setup" in payload:
        logger.info("[db_setup] 开始写入前置数据")
        conn = get_db()
        try:
            write_db_setup(conn, payload["db_setup"])
            conn.commit()
            logger.info("[db_setup] 写入完成")
        finally:
            conn.close()
    else:
        logger.info("[db_setup] 无前置数据，跳过")

    # ── 解析业务 XML ──
    raw_xml = payload["request"]
    logger.debug("[业务XML]\n%s", raw_xml)
    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as e:
        logger.warning("[XML解析失败] %s", e)
        return xml_resp("9999", error_msg="request 字段 XML 解析失败")

    user_id  = (root.findtext("user_id")  or "").strip()
    model_id = (root.findtext("model_id") or "").strip()
    channel  = (root.findtext("channel")  or "").strip()
    logger.info("[XML解析] user_id='%s'  model_id='%s'  channel='%s'",
                user_id, model_id, channel)

    if not user_id:
        logger.warning("[校验] user_id 为空")
        return xml_resp("9001", error_msg="user_id不能为空")
    if model_id != "MODEL_001":
        logger.warning("[校验] 不支持的 model_id='%s'", model_id)
        return xml_resp("9002", error_msg=f"不支持的model_id: {model_id}")

    # ── 查询 DB ──
    logger.info("[DB查询] user_id='%s'", user_id)
    conn = get_db()
    try:
        ui  = conn.execute("SELECT user_level          FROM user_info            WHERE user_id=?", (user_id,)).fetchone()
        ab  = conn.execute("SELECT avg_3m_balance       FROM account_balance      WHERE user_id=?", (user_id,)).fetchone()
        ss  = conn.execute("SELECT social_security_flag FROM cgs_social_security  WHERE user_id=?", (user_id,)).fetchone()
        sal = conn.execute("SELECT monthly_salary        FROM salary_summary       WHERE user_id=?", (user_id,)).fetchone()
    finally:
        conn.close()

    if not all([ui, ab, ss, sal]):
        missing = [t for t, v in [("user_info", ui), ("account_balance", ab),
                                   ("cgs_social_security", ss), ("salary_summary", sal)] if not v]
        logger.warning("[DB查询] 数据缺失，缺失表: %s", missing)
        return xml_resp("9003", error_msg=f"用户 {user_id} 数据不完整或不存在")

    user_level           = ui["user_level"]
    avg_3m_balance       = ab["avg_3m_balance"]
    social_security_flag = ss["social_security_flag"]
    monthly_salary       = sal["monthly_salary"]

    logger.info("[DB查询] user_level=%s  avg_3m_balance=%s  "
                "social_security_flag=%s  monthly_salary=%s",
                user_level, avg_3m_balance, social_security_flag, monthly_salary)

    # ── 业务逻辑 ──
    admit_cond = monthly_salary > 10000 and social_security_flag == 1
    logger.info("[准入判断] monthly_salary(%s) > 10000 AND social_security_flag(%s) == 1  =>  %s",
                monthly_salary, social_security_flag, admit_cond)

    if admit_cond:
        admit_flag   = 1
        raw_limit    = user_level * avg_3m_balance * monthly_salary * 0.0023
        credit_limit = max(raw_limit, 0.0)
        logger.info("[额度计算] %s × %s × %s × 0.0023 = %s  =>  归零后 %s",
                    user_level, avg_3m_balance, monthly_salary, raw_limit, credit_limit)
    else:
        admit_flag   = 0
        credit_limit = 0.0
        logger.info("[额度计算] 不准入，credit_limit 强制为 0.00")

    return xml_resp("0000", admit_flag=admit_flag, credit_limit=credit_limit)


# ──────────────────────────── 测试执行接口 ────────────────────────────

@app.route("/run-tests", methods=["POST"])
def run_tests():
    """
    接收 YAML 内容或文件路径，执行测试用例，返回 JSON 报告。

    请求体（JSON）：
      { "yaml_content": "..." }   直接传 YAML 文本
      { "yaml_path": "..." }      传文件路径（相对于服务器工作目录）
    """
    from yaml_runner import run_yaml_content, run_yaml_file

    body = request.get_json(silent=True) or {}
    server_url = body.get("server_url", "http://127.0.0.1:5000")

    if "yaml_content" in body:
        report = run_yaml_content(body["yaml_content"], server_url)
    elif "yaml_path" in body:
        report = run_yaml_file(body["yaml_path"], server_url)
    else:
        return jsonify({"error": "请提供 yaml_content 或 yaml_path"}), 400

    return jsonify(report)


# ──────────────────────────── 启动 ────────────────────────────

if __name__ == "__main__":
    init_db()
    logger.info("=" * 50)
    logger.info("  被测系统已启动: http://127.0.0.1:5000")
    logger.info("  接口: POST /api/model/score  (application/json)")
    logger.info("  日志文件: %s", LOG_FILE)
    logger.info("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
