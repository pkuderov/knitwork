
from aim import Run


def create_logger(config):
    cfg_log = config.get('log', dict())
    if not cfg_log.get('enabled', False):
        return None

    match cfg_log['logger']:
        case 'aim':
            return get_aim_logger(config)

    return None


def get_aim_logger(config):
    cfg_log = config['log']

    run = Run(
        experiment=cfg_log['project']
    )
    print(f'Logging to Aim: {run.hash} {cfg_log["project"]}')
    run['hparams'] = config | dict(
        project=cfg_log['project'],
    )
    return run

