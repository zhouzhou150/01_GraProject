from __future__ import annotations

from dataclasses import dataclass


WORKFLOW_STEP_TITLES = (
    "导入音频与参考文本",
    "选择模型并完成加载",
    "运行性能测试与总体测试",
    "查看结论并导出报告",
)


@dataclass(slots=True)
class WorkflowProgress:
    progress_value: float
    completed_steps: tuple[bool, bool, bool, bool]
    current_step: str


def compute_workflow_progress(
    dataset_ready: bool,
    loaded_model_count: int,
    performance_ready: bool,
    overall_ready: bool,
) -> WorkflowProgress:
    step_one = bool(dataset_ready)
    step_two = step_one and loaded_model_count > 0
    step_three = step_two and bool(performance_ready)
    step_four = step_three and bool(overall_ready)
    completed_steps = (step_one, step_two, step_three, step_four)

    if not step_one:
        current_step = WORKFLOW_STEP_TITLES[0]
    elif not step_two:
        current_step = WORKFLOW_STEP_TITLES[1]
    elif not step_three:
        current_step = WORKFLOW_STEP_TITLES[2]
    elif not step_four:
        current_step = WORKFLOW_STEP_TITLES[3]
    else:
        current_step = "评测流程已完成"

    return WorkflowProgress(
        progress_value=sum(completed_steps) / len(completed_steps),
        completed_steps=completed_steps,
        current_step=current_step,
    )
