import subprocess
import sys


BASE_CONFIG = {
    'script': 'train_seq_fusion.py',
    'mode': 'CAF',
    'epochs': 100,
    'lr': 1e-5,
    'dim': 256,
    'depth': 6,
    'heads': 4,
    'mlp_dim': 256,
    'dropout': 0.5,
    'batch_size': 32,
    'use_group_split': True,
    'use_sg_filter': True,
    'use_gated_fusion': True,
    'use_post_fusion_mlp': True,
    'use_gate_weight_feature': True,
    'input_branch': 'both',
}


# 做消融实验时，只需要把想跑的实验 enabled 改成 True。
# 每个实验会覆盖 BASE_CONFIG 里的同名参数，并保存到不同的结果文件。
ABLATION_EXPERIMENTS = [
    {
        'run_name': 'baseline_all_modules',
        'enabled': True,
    },
    {
        'run_name': 'ablation_vit_no_caf',
        'enabled': False,
        'mode': 'ViT',
    },
    {
        'run_name': 'ablation_no_sg_filter',
        'enabled': False,
        'use_sg_filter': False,
    },
    {
        'run_name': 'ablation_no_gated_fusion',
        'enabled': False,
        'use_gated_fusion': False,
    },
    {
        'run_name': 'ablation_no_post_fusion_mlp',
        'enabled': False,
        'use_post_fusion_mlp': False,
    },
    {
        'run_name': 'ablation_no_gate_weight_feature',
        'enabled': False,
        'use_gate_weight_feature': False,
    },
    {
        'run_name': 'ablation_leaf_only',
        'enabled': False,
        'input_branch': 'leaf',
    },
    {
        'run_name': 'ablation_soil_only',
        'enabled': False,
        'input_branch': 'soil',
    },
]


def build_command(params):
    script_to_run = params.get('script', 'train_seq_fusion.py')
    command = [sys.executable, script_to_run]

    for key, value in params.items():
        if key in {'script', 'enabled'} or value is None:
            continue

        if isinstance(value, bool):
            command.append(f'--{key}' if value else f'--no-{key}')
        else:
            command.extend([f'--{key}', str(value)])

    return command


def main():
    experiments = []
    for experiment in ABLATION_EXPERIMENTS:
        if not experiment.get('enabled', False):
            continue
        params = BASE_CONFIG.copy()
        params.update(experiment)
        experiments.append(params)

    if not experiments:
        print("没有启用任何实验，请在 ABLATION_EXPERIMENTS 中把 enabled 改为 True。")
        return

    for params in experiments:
        print("\n" + "="*50)
        print(f"Running with parameters: {params}")
        print("="*50)

        command = build_command(params)
        print("Command:", " ".join(command))
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with parameters: {params}")
            print(f"Error: {e}")
        except FileNotFoundError:
            print(f"Error: Could not find '{params.get('script', 'train_seq_fusion.py')}'. Make sure it's in the same directory.")
            break

if __name__ == '__main__':
    main()
