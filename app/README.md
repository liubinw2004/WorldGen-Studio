# app/ — 图形界面

Flask + 原生前端，无打包步骤、无前端依赖。入口是 `../start.bat`。

| 文件 | 作用 |
|---|---|
| `server.py` | HTTP 路由：生成、取消、上传、历史、SSE 事件流、GPU 遥测 |
| `runner.py` | 推理子进程调度：事件缓冲、日志解析、无输出看门狗、完成后回收 |
| `static/` | 单页界面（`index.html` / `style.css` / `app.js`） |
| `tools/cdp.py` | 通过 Chrome DevTools Protocol 截图与执行 JS（调试用） |

## 设计要点

**一次只跑一个任务。** 12GB 卡上峰值约 10.9GB，两个任务并行必然卡死，所以 runner 用互斥的方式串行调度，并在新任务开始前回收上一个仍在服务查看器的子进程。

**事件缓冲无锁。** `Job` 用固定长度 deque 加单调计数器；`deque.append` 与 `int += 1` 在 GIL 下是原子的。早期版本给每次写入加锁，结果 `set_stage` 持锁后又调用 `push` 重入同一把不可重入锁，日志管道冻结、子进程 stdout 写满、GPU 停在 100% 且永不退出。

**子进程写文件而非管道。** 管道有背压：读取端一旦停顿，子进程会卡在 `print()` 里。所以子进程把日志写进 `run.log`，pump 线程按 stat/seek 增量读取，并按 `\r\n|\r|\n` 切分以接住 tqdm 的进度条重绘。

**「进程还在」不等于「任务还在跑」。** `demo.py` 生成完成后会继续用 viser 在 :8080 提供查看器，永不退出。完成信号取自日志里的成功标记，而不是进程退出。

## 接口

| 路由 | 说明 |
|---|---|
| `GET /` | 界面 |
| `POST /api/generate` | `{mode, prompt, image, resolution, sr}` |
| `POST /api/cancel` | 终止当前任务 |
| `GET /api/events?since=N` | SSE 事件流（stage / progress / log / done） |
| `GET /api/state` | 任务快照 + GPU 遥测 |
| `GET /api/system` | 环境、模型就绪状态、超分模型 |
| `GET /api/history` | 场景库 |
| `POST /api/upload` | 参考图上传 |

完整说明见仓库根目录的 [README](../README.md)。
