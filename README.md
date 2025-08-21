# LLM 学习项目

这是一个专注于大语言模型学习和实践的代码仓库。

## 项目结构

```
LLM_learning/
├── Agent/                 # AI智能体相关代码
│   └── demo1/            # 基础AI智能体示例
│       ├── main.py       # 主程序入口
│       ├── tools.py      # 工具函数模块
│       └── ai_agent_analysis.md  # 代码原理详解
└── README.md             # 项目说明文档
```

## 当前功能

### Agent Demo1
- 🤖 基于 Ollama + LLaMA 3.2 的本地AI智能体
- 📁 文件操作能力（列表、读取、重命名）
- 💬 交互式对话界面
- 🔧 模块化工具系统

## 快速开始

1. 安装依赖
```bash
pip install pydantic-ai
```

2. 启动 Ollama 服务并拉取模型
```bash
ollama pull llama3.2
ollama serve
```

3. 运行AI智能体
```bash
cd Agent/demo1
python main.py
```

## 学习资源

- 📖 [AI Agent 代码原理详解](Agent/demo1/ai_agent_analysis.md)

## 技术栈

- **框架**: pydantic-ai
- **模型**: LLaMA 3.2 (通过 Ollama)
- **语言**: Python 3.x

---

*持续更新中...* 🚀
