import json
import os
import sys
import glob
from pathlib import Path
import numpy as np
from datetime import datetime

def sse_print(event: str, data: dict, progress: int = None, message: str = None, log: str = None, 
              callback_params: dict = None, details: dict = None) -> str:
    """
    SSE print with new standardized format:
    event: event_name
    data: {"callback_params": {...}, "progress": ..., "message": "...", "log": "...", ...}
    """
    # Initialize the payload (the content of the 'data:' line)
    payload = {}
    
    # Add callback_params if provided
    if callback_params:
        payload["callback_params"] = callback_params
    else:
        # Default callback params if none provided (as seen in user example)
        payload["callback_params"] = {
            "task_run_id": f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "method_type": "模型训练",
            "algorithm_type": "深度学习",
            "task_type": "决策式AI训练",
            "task_name": "决策式AI模型开发优化",
            "user_name": "admin"
        }
    
    # Add progress if provided
    if progress is not None:
        payload["progress"] = progress
    
    # Add message if provided
    if message:
        payload["message"] = message
    
    # Add log if provided
    if log:
        payload["log"] = log
    elif progress is not None and message:
        payload["log"] = f"[{progress}%] {message}\n"
    
    # Add details or merge with data
    if details:
        payload["details"] = details
    elif data and event not in ["input_path_validated", "output_path_validated", 
                                  "input_data_validated", "input_model_validated"]:
        payload["details"] = data
    else:
        # For validation events or other specific cases, merge data directly into payload
        payload.update(data)
    
    # Serialize the payload
    json_str = json.dumps(payload, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    
    # Construct the SSE multi-line output
    message_str = f"event: {event}\ndata: {json_str}\n\n"
    
    sys.stdout.write(message_str)
    sys.stdout.flush()
    return message_str

def sse_heartbeat(progress, message, callback_params=None):
    sse_print("progress_update", {}, progress=progress, message=message, callback_params=callback_params)

def sse_input_path_validated(args):
    try:
        if os.path.exists(args.input_path):
            sse_print("input_path_validated", {
                "status": "success",
                "message": "输入路径验证成功",
                "file_name": args.input_path
            }, progress=5, message="输入路径验证成功")
            
            try:
                found_data = False
                for ds in ['CIFAR-10', 'imagenet', 'MNIST']:
                    if os.path.exists(f'{args.input_path}/{ds}'):
                        sse_print("input_data_validated", {
                            "status": "success",
                            "message": f"输入数据文件验证成功 ({ds})",
                            "file_name": f'{args.input_path}/{ds}'
                        }, progress=10, message=f"输入数据验证成功 ({ds})")
                        found_data = True
                        break
                if not found_data:
                    raise ValueError('输入数据文件未找到 (CIFAR-10, imagenet, 或 MNIST)')
            except Exception as e:
                sse_print("input_data_validated", {"status": "failure", "message": f"{e}"})
                
            try:
                if os.path.exists(f'{args.input_path}/model'):
                    model_files = glob.glob(os.path.join(f'{args.input_path}/model', '*'))
                    sse_print("input_model_validated", {
                        "status": "success",
                        "message": "输入模型文件验证成功",
                        "file_name": model_files[0] if model_files else f'{args.input_path}/model'
                    }, progress=15, message="输入模型验证成功")
                else:
                    raise ValueError('输入模型文件未找到')
            except Exception as e:
                sse_print("input_model_validated", {"status": "failure", "message": f"{e}"})
        else:
            raise ValueError('输入路径未找到')
    except Exception as e:
        sse_print("input_path_validated", {"status": "failure", "message": f"{e}"})

def sse_output_path_validated(args):
    try:
        if os.path.exists(args.output_path):
            sse_print("output_path_validated", {
                "status": "success",
                "message": "输出路径验证成功",
                "file_name": args.output_path
            }, progress=20, message="输出路径验证成功")
        else:
            raise ValueError('输出路径未找到')
    except Exception as e:
        sse_print("output_path_validated", {"status": "failure", "message": f"{e}"})

def sse_epoch_progress(progress, total, epoch_type="Epoch"):
    progress_pct = int((progress / total) * 100)
    sse_print("training_progress", {
        "progress": progress,
        "total": total,
        "type": epoch_type
    }, progress=progress_pct, message=f"{epoch_type} {progress}/{total}")

def sse_error(message, event_name="error"):
    sse_print(event_name, {"status": "failure", "message": message})

def save_json_results(results: dict, output_path: str, filename: str = "results.json"):
    os.makedirs(output_path, exist_ok=True)
    json_path = os.path.join(output_path, filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda obj: obj.item() if isinstance(obj, np.generic) else float(obj) if isinstance(obj, (np.floating, np.integer)) else obj)
    return json_path

