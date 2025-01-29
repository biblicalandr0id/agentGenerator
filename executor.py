#executor.py
import torch
import torch.optim as optim
import numpy as np

class AdaptiveExecutor:
    def __init__(self, network,
                 learning_rates=[1e-3, 1e-4, 1e-5],
                 adaptation_strategies=['adam', 'sgd', 'rmsprop'],
                 adaptation_threshold=0.1,
                 lr_decay_factor=0.9,
                 lr_decay_patience=5
                ):
        self.network = network
        self.optimizer_classes = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }

        self.optimizers = []
        self.lr_schedulers = []
        for strategy in adaptation_strategies:
            for lr in learning_rates:
                optimizer_kwargs = {
                    'params': network.parameters(),
                    'lr': lr,
                    'weight_decay': 1e-5
                }
                if strategy == 'sgd':
                    optimizer_kwargs['momentum'] = 0.9

                optimizer_class = self.optimizer_classes[strategy]
                optimizer = optimizer_class(**optimizer_kwargs)
                self.optimizers.append(optimizer)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=lr_decay_factor, patience=lr_decay_patience, verbose=False, mode='min'
                )
                self.lr_schedulers.append(scheduler)

        self.current_optimizer_index = 0
        self.adaptation_threshold = adaptation_threshold

        self.performance_metrics = {
            'loss_history': [],
            'validation_loss_history': [],
            'gradient_norms': [],
            'optimizer_scores': [0] * len(self.optimizers),
            'strategy_performance': {
                strategy: {'total_score': 0, 'iterations': 0}
                for strategy in adaptation_strategies
            }
        }
        self.learning_rates = {
            (strategy, lr): lr
            for strategy in adaptation_strategies
            for lr in learning_rates
         }
        self.optimizer_switch_counter = 0

    def execute(self, inputs, targets, context, auxiliary_loss=None, diagnostics=None):
        current_optimizer = self.optimizers[self.current_optimizer_index]
        current_optimizer.zero_grad()

        outputs, state_importance = self.network(inputs, context)

        criterion = nn.MSELoss()
        primary_loss = criterion(outputs, targets)

        total_loss = primary_loss
        if auxiliary_loss is not None:
            total_loss += auxiliary_loss

        total_loss.backward()

        total_grad_norm = self._compute_gradient_norm()

        for param in self.network.parameters():
            if param.grad is not None:
                param.grad *= state_importance.mean() * (1 / (total_grad_norm + 1e-8))

        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        current_optimizer.step()

        loss_item = total_loss.item()
        self._update_performance(loss_item, total_grad_norm)

        current_scheduler = self.lr_schedulers[self.current_optimizer_index]
        validation_loss = self.performance_metrics['validation_loss_history'][-1] if self.performance_metrics['validation_loss_history'] else loss_item
        current_scheduler.step(validation_loss)

        return loss_item, outputs, state_importance

    def record_validation_loss(self, validation_loss):
        self.performance_metrics['validation_loss_history'].append(validation_loss)

    def _compute_gradient_norm(self):
        total_norm = 0
        grad_details = []

        for p in self.network.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_details.append({
                    'name': p.name if hasattr(p, 'name') else 'unnamed',
                    'grad_norm': param_norm.item()
                })

        total_norm = np.sqrt(total_norm)
        self.performance_metrics['gradient_norms'].append(total_norm)

        return total_norm

    def _update_performance(self, current_loss, grad_norm):
        self.performance_metrics['loss_history'].append(current_loss)

        performance_delta = (
            self.performance_metrics['loss_history'][-2] - current_loss
            if len(self.performance_metrics['loss_history']) > 1 else 0
        )

        current_strategy = list(self.optimizer_classes.keys())[
            self.current_optimizer_index // len(self.optimizers)//len(self.optimizer_classes)
        ]
        strategy_perf = self.performance_metrics['strategy_performance'][current_strategy]
        strategy_perf['total_score'] += performance_delta
        strategy_perf['iterations'] += 1

        if len(self.performance_metrics['loss_history']) % 50 == 0:
            strategy_scores_avg = {
                strategy: perf['total_score'] / (perf['iterations'] + 1e-9)
                for strategy, perf in self.performance_metrics['strategy_performance'].items()
            }

            best_strategy = max(strategy_scores_avg, key=strategy_scores_avg.get)
            best_index_base = list(self.optimizer_classes.keys()).index(best_strategy)

            current_avg_score = strategy_scores_avg[current_strategy]
            best_avg_score = strategy_scores_avg[best_strategy]

            if best_avg_score > current_avg_score * 1.02 and self.optimizer_switch_counter >= 3:
                 best_index = best_index_base * len(self.learning_rates)
                 self.current_optimizer_index = best_index
                 self.optimizer_switch_counter = 0
                 print(f"Optimizer switched to: {best_strategy} at index {self.current_optimizer_index}, Avg Scores: {strategy_scores_avg}")
            else:
                self.optimizer_switch_counter += 1

    def adjust_learning_rate(self, anomalies):
        if 'gradient_norm' in anomalies and anomalies['gradient_norm']:
            if np.any([a['z_score'] > 3 for a in anomalies['gradient_norm']]):
                current_optimizer = self.optimizers[self.current_optimizer_index]
                for param_group in current_optimizer.param_groups:
                    param_group['lr'] *= 0.75
                    print(f"Gradient Anomaly Detected: Learning rate reduced for optimizer {self.current_optimizer_index} to {param_group['lr']}")