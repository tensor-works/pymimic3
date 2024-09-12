import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchmetrics import Accuracy
from metrics.pytorch.wrappers import RunningAvgLoss, RunningAvgMetric


def test_running_avg_wrapper():
    # Set up the accuracy metric
    accuracy_metric = Accuracy(task="multiclass", num_classes=3)
    running_avg_accuracy = RunningAvgMetric(accuracy_metric)
    acc_total_slearn = 0.0

    # Simulate 5 batches of predictions and labels
    for batch in range(5):
        # Generate random predictions and labels
        inputs = torch.randn(10, 3)  # B, N
        targets = torch.randint(0, 3, (10,))  # B

        # Update the running average metric
        running_avg_accuracy.update(inputs, targets)

        sklearn_inputs = torch.argmax(inputs, axis=-1).numpy()
        sklearn_targets = targets.numpy()

        acc_total_slearn += accuracy_score(sklearn_targets, sklearn_inputs)

        # Print the current state
        print(f"Batch {batch + 1}:")
        print(f"  Sklearn Accuracy: {accuracy_score(sklearn_targets, sklearn_inputs):.4f}")
        print(f"  Running Average Accuracy (sklearn): {acc_total_slearn / (batch + 1):.4f}")
        print(f"  Batch Accuracy: {accuracy_metric.compute().item():.4f}")
        print(f"  Running Average Accuracy: {running_avg_accuracy.get():.4f}")
        print()

    # Final results
    print("Final Results:")
    print(f"  Torchmetrics Accuracy: {accuracy_metric.compute().item():.4f}")
    print(f"  Running Average Accuracy: {running_avg_accuracy.get():.4f}")


if __name__ == "__main__":
    test_running_avg_wrapper()
