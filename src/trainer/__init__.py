# Author: Bingxin Ke
# Last modified: 2024-05-17

from .metricgold_trainer import MetricgoldTrainer


trainer_cls_name_dict = {
    "MetricgoldTrainer": MetricgoldTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
