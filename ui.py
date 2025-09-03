import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import subprocess
import requests
import io
import os

class FedLearningUI:
    def __init__(self, root):
        self.root = root
        self.root.title("联邦学习训练平台")
        self.root.geometry("700x650")
        self.root.configure(bg="white")

        # === Logo ===
        url = "https://img0.baidu.com/it/u=863996299,3336643075&fm=253&fmt=auto&app=138&f=PNG?w=224&h=224"
        response = requests.get(url)
        img_data = response.content
        pil_img = Image.open(io.BytesIO(img_data)).resize((80, 80))
        self.logo_img = ImageTk.PhotoImage(pil_img)
        logo_label = tk.Label(root, image=self.logo_img, bg="white")
        logo_label.pack(pady=10)

        # === 算法模式选择 ===
        frame1 = tk.Frame(root, bg="white")
        frame1.pack(pady=5)
        tk.Label(frame1, text="算法模式:", bg="white").pack(side=tk.LEFT, padx=5)
        self.mode_var = ttk.Combobox(frame1, values=["sync", "async"], state="readonly")
        self.mode_var.current(0)
        self.mode_var.bind("<<ComboboxSelected>>", self.toggle_dir_input)
        self.mode_var.pack(side=tk.LEFT)

        # === 数据集选择 ===
        frame2 = tk.Frame(root, bg="white")
        frame2.pack(pady=5)
        tk.Label(frame2, text="数据集:", bg="white").pack(side=tk.LEFT, padx=5)
        self.dataset_var = ttk.Combobox(frame2, values=["mnist", "cifar"], state="readonly")
        self.dataset_var.current(1)
        self.dataset_var.pack(side=tk.LEFT)

        # === 参数输入框 ===
        frame3 = tk.Frame(root, bg="white")
        frame3.pack(pady=10)

        self.epochs_var = tk.StringVar(value="20")
        self.batchsize_var = tk.StringVar(value="64")
        self.lr_var = tk.StringVar(value="0.01")

        tk.Label(frame3, text="训练轮数:", bg="white").grid(row=0, column=0, padx=5, pady=2)
        tk.Entry(frame3, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(frame3, text="Batch大小:", bg="white").grid(row=1, column=0, padx=5, pady=2)
        tk.Entry(frame3, textvariable=self.batchsize_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        tk.Label(frame3, text="学习率:", bg="white").grid(row=2, column=0, padx=5, pady=2)
        tk.Entry(frame3, textvariable=self.lr_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # === 异步模式数据目录输入框 ===
        self.dir_frame = tk.Frame(root, bg="white")
        self.dir_label = tk.Label(self.dir_frame, text="数据目录 (--dir / --valdir):", bg="white")
        self.dir_var = tk.StringVar(value="./data_split/dataset_split_40")
        self.dir_entry = tk.Entry(self.dir_frame, textvariable=self.dir_var, width=40)
        self.dir_btn = tk.Button(self.dir_frame, text="浏览", command=self.browse_dir, bg="#2196F3", fg="white")

        # 默认隐藏
        self.dir_label.pack_forget()
        self.dir_entry.pack_forget()
        self.dir_btn.pack_forget()
        self.dir_frame.pack_forget()

        # === 开始按钮 ===
        self.start_btn = tk.Button(root, text="开始训练", command=self.start_training, bg="#4CAF50", fg="white", width=15)
        self.start_btn.pack(pady=15)

        # === 进度条 ===
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        self.progress_label = tk.Label(root, text="进度: 0%", bg="white")
        self.progress_label.pack()

        # === 日志输出窗口 ===
        log_frame = tk.Frame(root, bg="white")
        log_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        tk.Label(log_frame, text="训练日志:", bg="white").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=12, width=80, bg="#f4f4f4", fg="black", wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # === 底部说明 ===
        footer = tk.Label(root, text="本系统支持同步与异步联邦学习训练", font=("Arial", 9), fg="gray", bg="white")
        footer.pack(side=tk.BOTTOM, pady=5)

    def toggle_dir_input(self, event=None):
        """当选择异步模式时，显示目录输入框"""
        if self.mode_var.get() == "async":
            self.dir_frame.pack(pady=5)
            self.dir_label.pack(side=tk.LEFT, padx=5)
            self.dir_entry.pack(side=tk.LEFT, padx=5)
            self.dir_btn.pack(side=tk.LEFT, padx=5)
        else:
            self.dir_frame.pack_forget()

    def browse_dir(self):
        """选择目录"""
        selected_dir = filedialog.askdirectory(title="选择数据目录 (--dir / --valdir)")
        if selected_dir:
            self.dir_var.set(selected_dir)

    def start_training(self):
        mode = self.mode_var.get()
        dataset = self.dataset_var.get()
        epochs = int(self.epochs_var.get())
        batchsize = int(self.batchsize_var.get())
        lr = float(self.lr_var.get())
        data_dir = self.dir_var.get()

        # 清空日志
        self.log_text.delete(1.0, tk.END)

        # 保存参数
        self.total_epochs = epochs
        self.current_epoch = 0
        self.progress["value"] = 0
        self.progress_label.config(text="进度: 0%")

        # 子线程执行训练
        t = threading.Thread(target=self.run_training, args=(mode, dataset, epochs, batchsize, lr, data_dir))
        t.start()

        # 启动进度条更新
        self.update_progress()

    def run_training(self, mode, dataset, epochs, batchsize, lr, data_dir):
        try:
            if mode == "async":
                cmd = [
                    "python", "he.py",
                    "--mode", mode,
                    "--dataset", dataset,
                    "--epochs", str(epochs),
                    "--batchsize", str(batchsize),
                    "--lr", str(lr),
                    "--interval", "1",
                    "--dir", data_dir,
                    "--valdir", data_dir
                ]
            else:
                cmd = [
                    "python", "he.py",
                    "--mode", mode,
                    "--dataset", dataset,
                    "--epochs", str(epochs),
                    "--batchsize", str(batchsize),
                    "--lr", str(lr),
                    "--interval", "1"
                ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore"
            )

            for line in process.stdout:
                line = line.strip()
                print(line)
                self.log_text.insert(tk.END, line + "\n")
                self.log_text.see(tk.END)  # 滚动到最后
                if "Epoch" in line:
                    self.current_epoch += 1

            process.wait()

            # 训练完成
            self.show_result(mode)

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def update_progress(self):
        if self.current_epoch < self.total_epochs:
            percent = int((self.current_epoch / self.total_epochs) * 100)
            self.progress["value"] = percent
            self.progress_label.config(text=f"进度: {percent}%")
            self.root.after(500, self.update_progress)  # 0.5秒更新一次
        else:
            self.progress["value"] = 100
            self.progress_label.config(text="进度: 100%")

    def show_result(self, mode):
        img_path = f"./log/{mode}_training_loss.png"
        if os.path.exists(img_path):
            result_win = tk.Toplevel(self.root)
            result_win.title("训练结果")
            img = Image.open(img_path).resize((500, 300))
            img_tk = ImageTk.PhotoImage(img)
            label = tk.Label(result_win, image=img_tk)
            label.image = img_tk
            label.pack()
        else:
            messagebox.showinfo("提示", "未找到结果图像")

if __name__ == "__main__":
    root = tk.Tk()
    app = FedLearningUI(root)
    root.mainloop()

