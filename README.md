# 金融助手 —— 基于 MinerU 的财报 RAG 问答系统

针对财务 PDF（年报等）的本地 RAG（检索增强生成）问答系统。使用 [MinerU](https://github.com/opendatalab/MinerU) 对 PDF 做版面级智能解析，按"文本/表格"分别分块，经向量化后存入 ChromaDB，检索命中后由大模型生成回答。

## 整体流程

```
PDF 文件
   │  MinerU (read_fn + do_parse, pipeline 后端)
   ▼
content_list.json          ← 版面检测 + OCR + 表格转HTML + 公式转LaTeX
   │  _split_into_chunks
   ├─ 文本：按字数累积分块，标题强制切分
   └─ 表格：整表成块；大表按 <tr> 行切分并复制表头
   │  DashScope 嵌入接口（分批调用）
   ▼
ChromaDB 持久化向量库
   │  查询向量化 + 相似度检索 Top-5
   ▼
构建 Prompt → ChatOpenAI（OpenAI 兼容接口）→ 回答
```

## 项目结构

```
financial_assistant/
├── financial_assistant.py   # 主程序（解析、分块、向量化、检索、问答）
├── requirements.txt         # Python 依赖
├── .env                     # 密钥与网关配置（不入库，需自行创建）
├── my_pdfs/                 # 待解析的 PDF（不入库）
├── processed/               # MinerU 解析产物（不入库，自动生成）
└── my_chroma_db/            # ChromaDB 持久化目录（不入库，自动生成）
```

## 核心模块

| 类/方法 | 职责 |
|---|---|
| `MinerURAGSystem.process_documents` | 遍历 PDF 文件夹，调用 MinerU 解析为结构化 JSON |
| `MinerURAGSystem._split_into_chunks` | 文本/表格分别分块，保持表格结构完整 |
| `MinerURAGSystem._split_table_into_chunks` | 大 HTML 表格按行切分，每个分块复制表头 |
| `MinerURAGSystem.build_vector_store` | 分批向量化并写入 ChromaDB |
| `MinerURAGSystem.query_documents` | 查询向量化 + Top-K 相似度检索 |
| `RAGQASystem` | 构建财务分析师 Prompt，调用 LLM 生成回答 |

## 环境准备

1. Python 3.10+，安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   > MinerU 首次解析 PDF 时会自动下载版面检测、OCR、表格识别等模型（约 1GB+）到 `~/.cache/huggingface/hub`，国内网络可设置 `MINERU_MODEL_SOURCE=modelscope` 加速。

2. 在项目根目录创建 `.env`（参考 `.env.example`，密钥自行填写）：

   ```ini
   # 对话模型（OpenAI 兼容网关）
   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   LLM_MODEL=qwen-turbo
   LLM_API_KEY=sk-xxxx

   # 嵌入模型（DashScope 原生协议）
   EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/api/v1
   EMBEDDING_MODEL=text-embedding-v4
   EMBEDDING_API_KEY=sk-xxxx
   ```

   支持任意 OpenAI 兼容网关与 DashScope 协议网关，对话与嵌入可分别配置。

## 执行方式

```bash
conda activate ai        # 或你的目标环境
python financial_assistant.py
```

程序自动执行完整流程：

1. 检查 `processed/` 下是否已有解析结果，有则跳过解析（MinerU 解析较慢，且结果会复用）
2. 检查向量库是否已有数据，有则跳过向量化（避免重复消耗嵌入额度）
3. 对默认问题 `"2019年一季度归属于上市公司股东的净利润？"` 执行检索 + 问答，打印回答

修改 `__main__` 中的 `question` 变量即可提问其他问题。

## 注意事项

- `.env` 含密钥，已被 `.gitignore` 排除，切勿提交到仓库
- DashScope 各模型的免费额度独立计算，批量大小已按接口上限做了分批处理
- 首次运行完整流程耗时较长（MinerU 解析 + 模型下载），之后增量运行会自动跳过已完成的步骤
