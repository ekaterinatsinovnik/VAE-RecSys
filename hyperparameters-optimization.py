from clearml import Task
from clearml.automation import (DiscreteParameterRange,
                                HyperParameterOptimizer,
                                UniformIntegerParameterRange,
                                UniformParameterRange)
from clearml.automation.optuna import OptimizerOptuna


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print(f'Best objective reached {objective_value}')

def optimize():

    task = Task.init(project_name='VAE-RecSys',
                task_name='Optimizer',
                task_type=Task.TaskTypes.optimizer,
                reuse_last_task_id=False,
                auto_resource_monitoring=False)

    an_optimizer = HyperParameterOptimizer(
       
        base_task_id='5040a9a82fb44bbe81c842877fc2c088',
        hyper_parameters=[
            UniformIntegerParameterRange('General/epochs_num', min_value=50, max_value=500, step_size=50),
            DiscreteParameterRange('General/batch_size', values=[16, 32, 64, 128, 256, 512, 1024, 2048]),
            UniformParameterRange('General/weight_decay', min_value=1e-5, max_value=0.1, step_size=0.005),
            UniformParameterRange('General/learning_rate', min_value=1e-4, max_value=0.1, step_size=0.005),
            UniformParameterRange('General/dropout', min_value=0.01, max_value=0.6, step_size=0.01),
            UniformIntegerParameterRange('General/latent_dim', min_value=100, max_value=2000, step_size=50),
            UniformIntegerParameterRange('General/hidden_dim', min_value=100, max_value=2000, step_size=50),
            UniformIntegerParameterRange('General/num_hidden', min_value=0, max_value=6, step_size=1),
            # DiscreteParameterRange('General/set_lr_scheduler', values=[False, True]),
            ],
        
        objective_metric_title='Validation metric',
        objective_metric_series='NDCG@10',
        objective_metric_sign='max',
        optimizer_class=OptimizerOptuna,
        execution_queue='1xGPU',
        total_max_jobs=100000, # количество итераций общее
        save_top_k_tasks_only=10,
        max_number_of_concurrent_tasks=4,
        optimization_time_limit=480, 
        # compute_time_limit=30,  
        # min_iteration_per_job=150,  
        max_iteration_per_job=150000,
        auto_connect_task=True,
        # objective_metric
    )
    an_optimizer.set_report_period(5)

    an_optimizer.start_locally(job_complete_callback=job_complete_callback)

    top_exp = an_optimizer.get_top_experiments(top_k=5)
    print(top_exp)

    an_optimizer.wait()

    an_optimizer.stop()

if __name__ == "__main__":
    optimize()
