import tkinter as tk
from tkinter import messagebox
import yaml

def get_hyper_parameters():
    window_size, hidden_size, batch_size, max_epoch = 5, 100, 100, 10
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if config is None:
        messagebox.showwarning("警告", "配置文件为空，使用默认值")
    else:
        window_size = config.get('window_size', 0)
        hidden_size = config.get('hidden_size', 0)
        batch_size = config.get('batch_size', 0)
        max_epoch = config.get('max_epoch', 0)


    def submit():
        nonlocal window_size, hidden_size, batch_size, max_epoch
        window_size = int(entry_ws.get())
        hidden_size = int(entry_hs.get())
        batch_size = int(entry_bs.get())
        max_epoch = int(entry_me.get())
        config = {
            'window_size': window_size,
            'hidden_size': hidden_size,
            'batch_size': batch_size,
            'max_epoch': max_epoch
        }
        with open('config.yaml', 'w') as f:
            yaml.dump(config, f)
        root.destroy()

    def on_closing():
        root.destroy()

    root = tk.Tk()
    root.title("超参数设置")
    root.attributes('-topmost', True)  # 设置窗口置顶

    tk.Label(root, text="window_size").grid(row=0)
    tk.Label(root, text="hidden_size:").grid(row=1)
    tk.Label(root, text="batch_size:").grid(row=2)
    tk.Label(root, text="max_epoch:").grid(row=3)

    entry_ws = tk.Entry(root)
    entry_hs = tk.Entry(root)
    entry_bs = tk.Entry(root)
    entry_me = tk.Entry(root)
    
    entry_ws.insert(0, str(window_size))
    entry_hs.insert(0, str(hidden_size))
    entry_bs.insert(0, str(batch_size))
    entry_me.insert(0, str(max_epoch))

    entry_ws.grid(row=0, column=1)
    entry_hs.grid(row=1, column=1)
    entry_bs.grid(row=2, column=1)
    entry_me.grid(row=3, column=1)

    tk.Button(root, text='提交', command=submit).grid(row=4, column=1, pady=4)

    root.protocol("WM_DELETE_WINDOW", on_closing)  # 捕捉窗口关闭事件

    root.mainloop()

    return window_size, hidden_size, batch_size, max_epoch

# 调用函数并获取超参数
if __name__ == '__main__':
    window_size, hidden_size, batch_size, max_epoch = get_hyper_parameters()
    print(f"window_size: {window_size}, hidden_size: {hidden_size}, batch_size: {batch_size}, max_epoch: {max_epoch}") 