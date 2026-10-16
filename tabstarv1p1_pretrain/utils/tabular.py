from tabstarv1p1_pretrain.datasets.objects import SupervisedTask


def task_type_from_d_output(d_output: int) -> SupervisedTask:
    if d_output == 1:
        return SupervisedTask.REGRESSION
    elif d_output == 2:
        return SupervisedTask.BINARY
    elif d_output > 2:
        return SupervisedTask.MULTICLASS
    else:
        raise ValueError(f"Invalid d_output: {d_output}. Must be > 0.")