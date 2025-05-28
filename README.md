
# ETA: 事件树生成模型项目  
本项目基于金融文本的事件级情感分析任务（EFSA），通过构建事件树结构并结合大语言模型（LLM）实现事件与情感的联合检测。


## 一、项目概述  
- **目标**：从金融文本中提取四元组 `(公司, 粗粒度事件, 细粒度事件, 情感极性)`，通过事件树建模事件层次及语义关系。  
- **核心技术**：  
  - 事件树线性化（深度优先遍历）与序列生成；  
  - 基于 `ChatGLM3` 的微调模型，结合 LoRA 优化参数；  
  - 负对数似然损失函数优化结构化输出。  


## 二、目录结构  
```
ETA/
├─ data/          # 数据集（EFSA格式）
├─ model/         # 训练好的模型参数
├─ utils/         # 工具函数（如数据预处理、树结构操作）
├─ decoding.py    # 约束解码逻辑
├─ main.py        # 主程序（训练/推理入口）
└─ .idea/         # IDE配置文件
```


## 三、依赖环境  
- **框架**：PyTorch、Transformers  
- **模型**：ChatGLM3-6b-chat  
- **工具**：LoRA（低秩适应）、Adam优化器  


## 四、快速开始  
1. **安装依赖**  
   ```bash
   pip install torch transformers accelerate
   ```

2. **训练模型**  
   ```bash
   python main.py --train_data ./data/train.csv --model chatglm3-6b-chat --lora_rank 8
   ```

3. **推理示例**  
   ```python
   from main import EventTreeGenerator

   model = EventTreeGenerator()
   text = "奥飞娱乐涨停收盘，当日主力资金净流入1.08亿元..."
   output = model.generate(text)
   print(output)  # 输出：[(奥飞娱乐, 股票事务, 股价变动, 正面), ...]
   ```


## 五、项目亮点  
- **结构化建模**：通过事件树显式表达事件层次与情感关联，优于传统序列生成方法；  
- **领域适配**：针对金融文本优化，支持行业术语与复杂事件解析；  
- **可扩展性**：兼容其他LLM（如GPT-4、DeepSeek-v3），支持多模态数据扩展。  

 
