import datetime
import json
import os
from collections import defaultdict
from PIL import Image
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

replace_prompt = "Please answer yes or no."

def mot_doc_to_visual(doc):
    return [Image.open(image).convert("RGB") for image in doc["images"]]

def parse_pred_ans(pred_ans):
    """Brought from Otter Eval"""
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    elif len(pred_ans) == 1:
        if pred_ans == "y":
            pred_label = "yes"
        elif pred_ans == "n":
            pred_label = "no"
        else:
            pred_label = "other"
    else:
        prefix_pred_ans = pred_ans[:4]
        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label

def mot_doc_to_text(doc):
    question = doc["question"]
    return f"{question}"

def mot_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["answer"].lower().strip().replace(".", "")
    assert gt_ans in ["yes", "no"]
    assert pred_ans in ["yes", "no"]

    score = 1.0 if pred_ans == gt_ans else 0.0
    
    return {"accuracy": {"score": score}}