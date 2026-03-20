# TestAgent

> 基于 MCP 协议的智能 Agent 框架

## 项目简介

TestAgent 是一个模块化的智能 Agent 框架，采用 **MCP-aware Agent + 独立 MCP Server** 的分层架构。核心设计理念是将 Agent 能力与领域知识分离，通过可插拔的 Skill 系统支持多种应用场景。

## 架构概览

```
TestAgent/
├── Client/          # Python Agent 核心 + Node.js Server
│   ├── agent/      # ReActAgent 实现
│   ├── server/     # Fastify HTTP Server
│   └── .testagent/ # 配置、Skills、Agents
│
└── Server/          # MCP Server (TypeScript)
    └── src/        # MCP 工具实现
```

### 系统架构图

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面层                            │
│           Web 前端 / CLI / REST API                     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│            Node.js Server (Fastify)                     │
│   - Agent 进程管理                                       │
│   - SSE 消息流                                          │
│   - Socket.IO 实时推送                                   │
│   - 会话与存储管理                                       │
└───────────────────────┬─────────────────────────────────┘
                        │ subprocess
┌───────────────────────▼─────────────────────────────────┐
│           Python Agent Core (ReActAgent)                │
│   - 多 LLM 支持 (DashScope/OpenAI/Anthropic/Gemini)     │
│   - Hook 事件系统                                        │
│   - 多步骤规划                                          │
│   - 工具执行                                            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                    工具层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  基础工具    │  │  MCP Server  │  │   Skills     │  │
│  │ 文件/Shell   │  │  外部服务    │  │  领域工具    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 核心特性

### 多 LLM 提供商

| 提供商 | 状态 |
|--------|------|
| DashScope (阿里云) | 支持 |
| OpenAI | 支持 |
| Anthropic | 支持 |
| Google Gemini | 支持 |
| Ollama (本地) | 支持 |

### 三层工具体系

1. **基础工具** - 文件操作、Shell 执行、HTTP 请求
2. **MCP 工具** - 通过 MCP 协议连接的外部服务
3. **Skill 工具** - 可插拔的领域专业工具包

### 可扩展架构

- **Skills**: 领域知识封装，包含工具定义和文档
- **Agents**: 预定义的专业 Agent (如 planner)
- **MCP Servers**: 独立的工具服务器

## 快速开始

### 环境要求

- Python 3.12+
- Node.js 18+
- PostgreSQL (可选，用于持久化)
- Redis (可选，用于缓存)

### 安装

```bash
# 克隆仓库
git clone --recursive https://github.com/GUODnuli/TestAgent.git
cd TestAgent

# 安装 Client 依赖
cd Client
pip install -r requirements.txt

# 安装 Server 依赖 (Node.js)
cd server
npm install

# 安装 MCP Server 依赖
cd ../../Server
npm install
```

### 配置

```bash
# Client 配置
cd Client
cp .testagent/settings.example.json .testagent/settings.json
# 编辑 settings.json 配置 LLM 和 MCP 服务器

# MCP Server 配置
cd ../Server
cp mcp-settings.example.json mcp-settings.json
```
创建 .env 文件
```bash
cp .env.example .env
```
编辑 .env 文件，配置数据库连接和 Redis 缓存。
# 数据库连接字符串（修改成自己的）
DATABASE_URL=postgresql://mcp_user:mcp_password@localhost:5432/testagent

# JWT 密钥（可以随便设置一个字符串）
JWT_SECRET="your-secret-key-here"


### 运行

```bash
# 启动 Node.js Server
cd Client/server
npm run dev

# MCP Server 会通过配置自动启动
```

## 项目结构

```
TestAgent/
├── Client/                     # Agent 客户端
│   ├── agent/                 # Python Agent 核心
│   │   ├── main.py           # 入口点
│   │   ├── hook.py           # 事件推送
│   │   ├── model.py          # LLM 适配
│   │   ├── tool_registry.py  # 工具管理
│   │   ├── plan/             # 规划系统
│   │   └── tool/             # 工具实现
│   ├── server/               # Node.js Server
│   │   └── src/
│   │       ├── agent/        # 进程管理
│   │       └── modules/      # API 模块
│   ├── .testagent/           # 扩展配置
│   │   ├── settings.json     # 主配置
│   │   ├── agents/           # Agent 定义
│   │   ├── skills/           # 技能包
│   │   └── rules/            # 规则
│   ├── prompts/              # 系统提示词
│   ├── frontend/             # Vue 前端
│   └── storage/              # 数据存储
│
├── Server/                    # MCP Server
│   ├── src/                  # TypeScript 源码
│   └── package.json
│
└── README.md                  # 本文件
```

## 扩展开发

### 添加 Skill

在 `Client/.testagent/skills/` 创建新目录：

```
skills/your_skill/
├── SKILL.md          # 元数据与文档
└── tools/
    ├── __init__.py
    └── your_tools.py # 工具实现
```

### 添加 MCP Server

1. 实现 MCP Server
2. 在 `settings.json` 中配置

```json
{
  "mcpServers": {
    "your_server": {
      "command": "node",
      "args": ["path/to/server.js"],
      "enabled": true
    }
  }
}
```

### 添加 Agent

在 `Client/.testagent/agents/` 创建 markdown 文件：

```yaml
---
name: your-agent
description: Agent 描述
tools: Read, Grep, Glob
model: qwen3-max
---

Agent 系统提示词...
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | AgentScope |
| Python 后端 | Python 3.12 |
| Node.js Server | Fastify, Prisma, Socket.IO |
| MCP Server | @modelcontextprotocol/sdk |
| 数据库 | PostgreSQL, Redis |
| 向量数据库 | ChromaDB |

## 相关链接

- [Client 文档](./Client/README.md)
- [AgentScope](https://github.com/modelscope/agentscope)
- [Model Context Protocol](https://modelcontextprotocol.io/)

## 许可证

MIT License
