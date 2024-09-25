from celery import Celery
import os

def make_celery(app):
    # Initialise Celery avec le broker et backend Redis
    celery = Celery(
        app.import_name,
        backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'),
        broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
    )

    # Synchroniser la configuration Flask avec Celery
    celery.conf.update(app.config)

    # Définir les files d'attente pour les tâches liées aux paires de devises
    celery.conf.task_queues = {
        'currency_queue': {
            'exchange': 'currency_exchange',
            'routing_key': 'currency.#',
            'queue_arguments': {'x-max-priority': 10}  # File d'attente prioritaire
        }
    }

    # Définir un maximum de rétentions et des options supplémentaires
    celery.conf.task_default_queue = 'currency_queue'
    celery.conf.task_default_exchange = 'currency_exchange'
    celery.conf.task_default_routing_key = 'currency.#'

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery
