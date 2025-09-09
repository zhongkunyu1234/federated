import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, subprocess, os, sys, time, io, base64, urllib.request, re

# ----------------- 小工具：下载 logo 并转为 PhotoImage -----------------
def load_remote_logo(url, size=(80, 80)):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        # PhotoImage 支持 base64 字符串
        b64 = base64.b64encode(data).decode('ascii')
        img = tk.PhotoImage(data=b64)
        # 简易等比缩放到近似尺寸
        w, h = img.width(), img.height()
        sx = max(1, w // size[0])
        sy = max(1, h // size[1])
        img = img.subsample(sx, sy)
        return img
    except Exception:
        return None

# ----------------- 渐变背景画布 -----------------
class GradientCanvas(tk.Canvas):
    def __init__(self, master, top_color="#5AA9FF", bottom_color="#FFFFFF", **kwargs):
        super().__init__(master, highlightthickness=0, **kwargs)
        self.top_color = top_color
        self.bottom_color = bottom_color
        self.bind("<Configure>", self._draw_gradient)

    def _hex_to_rgb(self, hx):
        hx = hx.lstrip("#")
        return tuple(int(hx[i:i+2], 16) for i in (0, 2, 4))

    def _draw_gradient(self, event=None):
        self.delete("grad")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 0 or h <= 0: return
        r1, g1, b1 = self._hex_to_rgb(self.top_color)
        r2, g2, b2 = self._hex_to_rgb(self.bottom_color)
        steps = max(2, h)
        for i in range(steps):
            r = int(r1 + (r2 - r1) * i / steps)
            g = int(g1 + (g2 - g1) * i / steps)
            b = int(b1 + (b2 - b1) * i / steps)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.create_line(0, i, w, i, tags="grad", fill=color)

# ----------------- 页面容器（带滑动动画） -----------------
class Wizard(tk.Tk):
    def __init__(self, PIL_AVAILABLE=None):
        super().__init__()
        self.title("联邦学习平台")
        self.geometry("1000x780")
        self.minsize(820, 640)

        # 背景
        self.bg = GradientCanvas(self, top_color="#6EB5FF", bottom_color="#FFFFFF")
        self.bg.pack(fill=tk.BOTH, expand=True)

        # 顶部 Header（Logo + 标题）
        # 半透明 header 浮层
        header_h = 80
        self.header_canvas = tk.Canvas(self.bg, highlightthickness=0,bg='#7DBCFF',bd=0)
        self.bg.create_window(0, 0, window=self.header_canvas, anchor="nw", width=self.winfo_screenwidth(),
                              height=header_h, tags="header_layer")

        # 放置 Logo + 标题
        self.logo = load_remote_logo(
            "https://img0.baidu.com/it/u=863996299,3336643075&fm=253&fmt=auto&app=138&f=PNG?w=224&h=224",
            size=(56, 56))
        if self.logo:
            r = max(self.logo.height(), self.logo.width()) // 2 + 10  # 半径
            cx = 30 + self.logo.width() // 2  # 圆心 x
            cy = header_h // 2  # 圆心 y
            self.header_canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                           fill="#FFFFFF", outline="")  # 半透明白色
            self.header_canvas.create_image(30, header_h // 2, image=self.logo, anchor="w")
        self.header_canvas.create_text(400, header_h // 2, text="联邦学习平台",
                                       font=("Segoe UI", 24, "bold"),
                                       fill="#002D6B", anchor="w")

        # 内容容器
        self.card = tk.Frame(self.bg, bg="white", bd=0, highlightthickness=0)
        self.card.place(relx=0.5, rely=0.55, anchor="c", relwidth=0.9, relheight=0.85)
        self.card.configure(borderwidth=0)
        self.card.grid_propagate(False)

        # 页面帧容器
        self.stack = tk.Frame(self.card, bg="white")
        self.stack.place(relx=0.5, rely=0.5, anchor="c", relwidth=0.95, relheight=0.90)

        self.pages = []
        self.current_index = 0

        # 全局配置状态
        self.dataset_var = tk.StringVar(value="cifar")   # mnist / cifar / custom
        self.custom_dir_var = tk.StringVar(value="")
        self.goal_var = tk.StringVar(value="quick")       # quick / research
        self.epochs_var = tk.IntVar(value=100)              # 根据目标自动推荐
        self.mode_var = tk.StringVar(value="sync")        # sync / async
        self.async_dir_var = tk.StringVar(value="")
        self.batchsize_var = tk.IntVar(value=64)
        self.lr_var = tk.DoubleVar(value=0.01)

        # 训练状态
        self.total_epochs = 0
        self.current_epoch = 0
        self.proc = None
        self.log_buffer = []

        # 构建页面
        self._build_pages()
        self._show_page(0, animate=False)

    # ---------- 页面构建 ----------
    def _build_pages(self):
        self.pages = [
            self._build_page1(),
            self._build_page2(),
            self._build_page3(),
            self._build_page4(),
        ]

    # Page 1：选择数据类型
    def _build_page1(self):
        frame = tk.Frame(self.stack, bg="white")
        title = tk.Label(frame, text="欢迎使用联邦学习平台", font=("Microsoft YaHei UI", 20, "bold"), bg="white")
        subtitle = tk.Label(frame, text="让我们开始创建一个新的图像识别模型", font=("Microsoft YaHei UI", 12), bg="white", fg="#555")

        title.pack(pady=(10, 4))
        subtitle.pack(pady=(0, 18))

        sec = tk.Label(frame, text="请选择用于训练的数据类型", font=("Microsoft YaHei UI", 14, "bold"), bg="white")
        sec.pack(anchor="w", padx=20)

        opt = tk.Frame(frame, bg="white")
        opt.pack(fill="x", padx=20, pady=(8, 8))

        rb1 = ttk.Radiobutton(opt, text="MNIST（手写数字）", variable=self.dataset_var, value="mnist", command=self._toggle_custom_area)
        rb2 = ttk.Radiobutton(opt, text="CIFAR-10（通用彩色图像）", variable=self.dataset_var, value="cifar", command=self._toggle_custom_area)
        rb3 = ttk.Radiobutton(opt, text="自定义目录数据集（已准备好的图像数据）", variable=self.dataset_var, value="custom", command=self._toggle_custom_area)

        rb1.grid(row=0, column=0, sticky="w", pady=4)
        rb2.grid(row=1, column=0, sticky="w", pady=4)
        rb3.grid(row=2, column=0, sticky="w", pady=4)

        self.custom_area = tk.Frame(frame, bg="white")
        tk.Label(self.custom_area, text="数据目录（当选择自定义时作为 --dir/--valdir 使用）", bg="white").pack(anchor="w")
        dir_row = tk.Frame(self.custom_area, bg="white")
        dir_row.pack(fill="x", pady=4)
        entry = ttk.Entry(dir_row, textvariable=self.custom_dir_var, width=50)
        entry.pack(side="left", padx=(0,6))
        ttk.Button(dir_row, text="浏览", command=self._browse_custom_dir).pack(side="left")

        nav = tk.Frame(frame, bg="white")
        nav.pack(side="bottom", pady=8)
        ttk.Button(nav, text="下一步 ➜", command=lambda: self._next_page(1)).pack()

        return frame

    # Page 2：训练目标
    def _build_page2(self):
        frame = tk.Frame(self.stack, bg="white")
        title = tk.Label(frame, text="选择训练目标", font=("Microsoft YaHei UI", 20, "bold"), bg="white")
        subtitle = tk.Label(frame, text="我们会根据目标帮你预设合适的超参数，若你不满意仍可调整", font=("Microsoft YaHei UI", 12), bg="white", fg="#555")
        title.pack(pady=(10, 4))
        subtitle.pack(pady=(0, 18))

        opt = tk.Frame(frame, bg="white")
        opt.pack(fill="x", padx=20, pady=4)

        def on_goal_change():
            if self.goal_var.get() == "quick":
                self.epochs_var.set(100)
            else:
                self.epochs_var.set(1000)

        rb1 = ttk.Radiobutton(opt, text="快速验证（更快见效果，适合参数调试）", variable=self.goal_var, value="quick", command=on_goal_change)
        rb2 = ttk.Radiobutton(opt, text="正式研究（更充分训练，追求指标）", variable=self.goal_var, value="research", command=on_goal_change)
        rb1.grid(row=0, column=0, sticky="w", pady=4)
        rb2.grid(row=1, column=0, sticky="w", pady=4)

        grid = tk.Frame(frame, bg="white")
        grid.pack(padx=20, pady=10, anchor="w")

        ttk.Label(grid, text="训练轮数（epochs）").grid(row=0, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(grid, textvariable=self.epochs_var, width=10).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(grid, text="Batch Size").grid(row=1, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(grid, textvariable=self.batchsize_var, width=10).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(grid, text="Learning Rate").grid(row=2, column=0, sticky="w", padx=(0,8), pady=4)
        ttk.Entry(grid, textvariable=self.lr_var, width=10).grid(row=2, column=1, sticky="w", pady=4)

        nav = tk.Frame(frame, bg="white")
        nav.pack(side="bottom", pady=8)
        ttk.Button(nav, text="⬅ 上一步", command=lambda: self._next_page(0, back=True)).pack(side="left", padx=6)
        ttk.Button(nav, text="下一步 ➜", command=lambda: self._next_page(2)).pack(side="left", padx=6)

        return frame

    # Page 3：训练策略
    def _build_page3(self):
        frame = tk.Frame(self.stack, bg="white")
        title = tk.Label(frame, text="选择训练策略", font=("Microsoft YaHei UI", 20, "bold"), bg="white")
        subtitle = tk.Label(frame, text="不同策略在稳定性与时效性之间取舍不同，请按需选择", font=("Microsoft YaHei UI", 12), bg="white", fg="#555")
        title.pack(pady=(10, 4))
        subtitle.pack(pady=(0, 18))

        cards = tk.Frame(frame, bg="white")
        cards.pack(fill="x", padx=10)

        # 同步策略卡片
        card_sync = tk.Frame(cards, bg="#F7FBFF", highlightbackground="#D6E9FF", highlightthickness=1, padx=12, pady=10)
        tk.Label(card_sync, text="同步策略（推荐入门）", bg="#F7FBFF", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        tk.Label(card_sync, text="• 各客户端对齐后再聚合，易于稳定收敛\n• 速度偏慢，对慢客户端敏感", bg="#F7FBFF", justify="left").pack(anchor="w")
        ttk.Radiobutton(card_sync, text="选择同步训练", variable=self.mode_var, value="sync").pack(anchor="w", pady=(6,0))

        # 异步策略卡片
        card_async = tk.Frame(cards, bg="#FFF8F5", highlightbackground="#FFE1D5", highlightthickness=1, padx=12, pady=10)
        tk.Label(card_async, text="异步策略（推荐追求时效）", bg="#FFF8F5", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        tk.Label(card_async, text="• 客户端随到随聚合，吞吐高、等待少\n• 存在延迟/陈旧梯度，调参更关键", bg="#FFF8F5", justify="left").pack(anchor="w")
        ttk.Radiobutton(card_async, text="选择异步训练", variable=self.mode_var, value="async").pack(anchor="w", pady=(6,0))

        # 缓存辅助异步策略卡片
        card_ca2fl = tk.Frame(cards, bg="#FFF8F5", highlightbackground="#F6E473", highlightthickness=1, padx=12, pady=10)
        tk.Label(card_ca2fl, text="缓存辅助异步策略（比普通异步更高效）", bg="#FFF8F5", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        tk.Label(card_ca2fl, text="• 收敛速度快且稳定\n• 缓解数据异构性带来的收敛延迟", bg="#FFF8F5", justify="left").pack(anchor="w")
        ttk.Radiobutton(card_ca2fl, text="选择缓存辅助异步训练", variable=self.mode_var, value="ca2fl").pack(anchor="w", pady=(6,0))

        card_sync.pack(side="left", expand=True, fill="both", padx=(0,8))
        card_async.pack(side="left", expand=True, fill="both", padx=(8,0))
        card_ca2fl.pack(side="left", expand=True, fill="both", padx=(8,0))

        nav = tk.Frame(frame, bg="white")
        nav.pack(side="bottom", pady=8)
        ttk.Button(nav, text="⬅ 上一步", command=lambda: self._next_page(1, back=True)).pack(side="left", padx=6)
        ttk.Button(nav, text="开始训练 ➜", command=self._start_training).pack(side="left", padx=6)

        return frame

    # Page 4：训练过程展示
    def _build_page4(self):
        frame = tk.Frame(self.stack, bg="white")
        title = tk.Label(frame, text="训练过程", font=("Microsoft YaHei UI", 20, "bold"), bg="white")
        subtitle = tk.Label(frame, text="下方展示实时日志、进度与输出图表", font=("Microsoft YaHei UI", 12), bg="white", fg="#555")
        title.pack(pady=(10, 4))
        subtitle.pack(pady=(0, 10))

        # 进度条 + 百分比
        pb_row = tk.Frame(frame, bg="white")
        pb_row.pack(fill="x", padx=12, pady=8)  # 给左右留点间距，和日志对齐
        self.progress = ttk.Progressbar(pb_row, orient="horizontal", mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.progress_label = tk.Label(pb_row, text="0%", bg="white")
        self.progress_label.pack(side="right")

        # 日志
        log_frame = tk.Frame(frame, bg="white")
        log_frame.pack(fill="both", expand=True, padx=12, pady=(6, 6))
        tk.Label(log_frame, text="训练日志", bg="white").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=12, bg="#f6f6f6", wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

        # 输出图像区域
        out_frame = tk.Frame(frame, bg="white")
        out_frame.pack(fill="x", padx=12, pady=8)
        tk.Label(out_frame, text="输出图表", bg="white").pack(anchor="w")
        self.out_canvas = tk.Label(out_frame, bg="#fafafa", width=640, height=320, relief="groove")
        self.out_canvas.pack(fill="x", pady=6)

        return frame

    # ---------- 导航/动画 ----------
    def _next_page(self, idx, back=False):
        # 校验：自定义数据集时必须给目录
        if idx == 1:  # 进入第二页时，根据选择设置默认 epoch
            if self.dataset_var.get() == "custom" and not self.custom_dir_var.get().strip():
                messagebox.showwarning("提示", "请选择自定义数据集目录")
                return
        self._show_page(idx, animate=True, back=back)

    def _show_page(self, idx, animate=True, back=False):
        if idx < 0 or idx >= len(self.pages): return
        new_page = self.pages[idx]
        w = self.stack.winfo_width() or self.stack.winfo_reqwidth()
        h = self.stack.winfo_height() or self.stack.winfo_reqheight()
        for p in self.pages:
            p.place_forget()
        if not animate:
            new_page.place(x=0, y=0, relwidth=1, relheight=1)
        else:
            # 滑动动画
            start_x = -w if not back else w
            new_page.place(x=start_x, y=0, relwidth=1, relheight=1)
            cur_x = start_x
            def step():
                nonlocal cur_x
                target = 0
                cur_x += (target - cur_x) * 0.25  # ease-out
                if abs(cur_x - target) < 1:
                    new_page.place(x=0, y=0, relwidth=1, relheight=1)
                else:
                    new_page.place(x=int(cur_x), y=0, relwidth=1, relheight=1)
                    self.after(16, step)
            step()
        self.current_index = idx

    # ---------- 事件 ----------
    def _browse_custom_dir(self):
        d = filedialog.askdirectory(title="选择自定义数据集目录")
        if d:
            self.custom_dir_var.set(d)
    
    def _toggle_custom_area(self):
        # Page1 上显示自定义目录行
        if self.dataset_var.get() == "custom":
            self.custom_area.pack(fill="x", padx=20, pady=(6, 0))
        else:
            self.custom_area.pack_forget()

    # ---------- 开始训练 ----------
    def _start_training(self):
        # 汇总参数
        dataset = self.dataset_var.get()
        mode = self.mode_var.get()
        epochs = int(self.epochs_var.get())
        batchsize = int(self.batchsize_var.get())
        lr = float(self.lr_var.get())

        # 根据数据集自动模型
        if dataset == "mnist" :
            dir_path = "./DATASET_DIR_MNIST"
            val_dir_path = "./DATASET_DIR_MNIST"
        elif dataset == "cifar":
            dir_path = ".\DATASET_DIR_CIFAR10"
            val_dir_path = ".\DATASET_DIR_CIFAR10"
        elif dataset == "custom":
            dir_path = self.custom_dir_var.get()
            val_dir_path = self.custom_dir_var.get()

        # 构建命令
        if mode == "sync":
            cmd = ["python", "fedsync.py",
                "--classes", str(10),
                "--model", "default",
                "--nsplit", str(100),
                "--epochs", str(epochs),
                "--batchsize", str(batchsize),
                "--lr", str(lr),
                "--frac", str(0.1),  # 每轮选择10%的客户端
                "--local_epochs", str(1),  # 每个客户端本地训练1轮
                "--lr_decay", str(0.1),  # 学习率衰减率
                "--lr_decay_epoch", "50,80",  # 在第50和80轮衰减学习率
                "--min_clients", str(10),  # 每轮最少10个客户端
                "--aggregation", "fedavg",  # 使用FedAvg聚合
                "--seed", str(336),
                "--dir", dir_path,
                "--valdir", val_dir_path]

        elif mode == "async":
            cmd = ["python", "fedasync.py",
                "--classes", str(10),
                "--model", "default",
                "--nsplit", str(100),
                "--epochs", str(epochs),
                "--batchsize", str(batchsize),
                "--lr", str(lr),
                "--rho", str(0.01), 
                "--alpha", str(0.8),
                "--alpha-decay", str(0.5),
                "--alpha-decay-epoch", str(800), 
                "--max-delay", str(12), 
                "--iterations", str(1), 
                "--seed", str(336),
                "--dir", dir_path,
                "--valdir", val_dir_path]
        
        elif mode == "ca2fl":
            cmd = ["python", "fedca2fl.py",
                "--classes", str(10),
                "--model", "default",
                "--nsplit", str(100),
                "--epochs", str(epochs),
                "--batchsize", str(batchsize),
                "--lr", str(lr),
                "--eta", str(0.01),
                "--M", str(10),  
                "--iterations", str(1), 
                "--seed", str(336),
                "--dir", dir_path,
                "--valdir", val_dir_path]

        # 训练状态初始化
        self.total_epochs = epochs
        self.current_epoch = 0
        # 切到第 4 页
        self._show_page(3, animate=True)
        self._clear_log()
        self._append_log(f"启动训练：mode={mode}, dataset={dataset}, epochs={epochs}, batchsize={batchsize}, lr={lr}")

        # 开线程运行子进程
        t = threading.Thread(target=self._run_proc, args=(cmd, mode))
        t.daemon = True
        t.start()

        # 开启丝滑进度刷新
        self._smooth_progress_tick()

    # ---------- 子进程与日志 ----------
    def _run_proc(self, cmd, mode):
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )
            epoch_pat = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)

            self._append_log(self.proc.stdout.readline())
            self._append_log(self.proc.stdout.readline())
            line1 = self.proc.stdout.readline()
            self._append_log(line1)
            m = epoch_pat.search(line1)
            if m:
                self.current_epoch = min(self.current_epoch + 1, self.total_epochs)
            for line in self.proc.stdout:
                line = line.rstrip()
                self._append_log(line)
                # 解析 Epoch 进度
                m = epoch_pat.search(line)
                if m:
                    # 检测到一次就 +10 的策略
                    self.current_epoch = min(self.current_epoch + 10, self.total_epochs)

            self.proc.wait()
            code = self.proc.returncode
            if code == 0:
                self._append_log("训练完成 ✅")
                self._load_result_image()
                # 直接填满进度
                self.current_epoch = self.total_epochs
            else:
                self._append_log(f"训练进程非零退出，code={code} ❌")
                messagebox.showerror("训练失败", f"训练进程退出码：{code}")
        except Exception as e:
            self._append_log(f"[异常] {e}")
            messagebox.showerror("异常", str(e))
        finally:
            self.proc = None

    def _append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    # 丝滑进度：持续向目标值逼近
    def _smooth_progress_tick(self):
        target = 0 if self.total_epochs == 0 else (self.current_epoch / max(1, self.total_epochs)) * 100.0
        cur = self.progress["value"]
        # 每帧向目标靠近，避免“一跳一跳”
        cur += (target - cur) * 0.2
        if abs(cur - target) < 0.5:
            cur = target
        self.progress["value"] = max(0, min(100, cur))
        self.progress_label.config(text=f"{int(self.progress['value']):d}%")
        # 只要训练还没到 100 或子进程还活着，就继续刷新
        if self.progress["value"] < 100.0 or self.proc is not None:
            self.after(120, self._smooth_progress_tick)

        # ---------- 结果图展示 ----------
    def _load_result_image(self):
        img_path = f"./log/training_curves.png"
        if os.path.exists(img_path):
            try:
                # 使用 PIL 处理图像
                from PIL import Image, ImageTk
                pil_img = Image.open(img_path)
                # 获取容器尺寸
                canvas_width = self.out_canvas.winfo_width() or 640
                canvas_height = self.out_canvas.winfo_height() or 320
                
                # 计算缩放比例，保持宽高比
                img_width, img_height = pil_img.size
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # 缩放图像
                pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                img = ImageTk.PhotoImage(pil_img)
                
                self.out_canvas.configure(image=img)
                self.out_canvas.image = img  # 保持引用
                self._append_log(f"结果图已加载：{img_path}")
                return
                
            except Exception as e:
                self._append_log(f"加载结果图失败：{e}")
        self._append_log("未找到结果图像，检查 ./log/ 目录。")

# ----------------- 入口 -----------------
if __name__ == "__main__":
    app = Wizard()
    app.mainloop()


