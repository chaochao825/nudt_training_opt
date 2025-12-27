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
    SSE print with standardized format
    """
    response = {
        "resp_code": 0,
        "resp_msg": "操作成功",
        "time_stamp": datetime.now().strftime("%Y/%m/%d-%H:%M:%S:%f")[:-3],
        "data": {
            "event": event
        }
    }
    
    if callback_params:
        response["data"]["callback_params"] = callback_params
    
    if progress is not None:
        response["data"]["progress"] = progress
    
    if message:
        response["data"]["message"] = message
    
    if log:
        response["data"]["log"] = log
    elif progress is not None and message:
        response["data"]["log"] = f"[{progress}%] {message}\n"
    
    if details:
        response["data"]["details"] = details
    elif data and event not in ["input_path_validated", "output_path_validated", 
                                  "input_data_validated", "input_model_validated"]:
        response["data"]["details"] = data
    else:
        response["data"].update(data)
    
    json_str = json.dumps(response, ensure_ascii=False, default=lambda obj: obj.item() if isinstance(obj, np.generic) else obj)
    message_str = f"data: {json_str}\n\n"
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

