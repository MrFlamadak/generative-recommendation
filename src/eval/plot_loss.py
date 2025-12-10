import csv
import matplotlib.pyplot as plt

def plot_train_val_loss():
    steps, losses = [], []
    with open('bart-recommender/train_losses.csv') as f:
        rows = csv.DictReader(f)
        for row in rows:
            steps.append(int(row['step']) if row['step'] is not None else None)
            losses.append(float(row['loss']) if row['loss'] is not None else None)
        plt.plot(steps, losses, label='train_loss')

    steps, val_losses = [], []
    with open('bart-recommender/eval_losses.csv') as f:
        rows = csv.DictReader(f)
        for row in rows:
            steps.append(int(row['step']) if row['step'] is not None else None)
            val_losses.append(float(row['eval_loss']) if row['eval_loss'] is not None else None)
        plt.plot(steps, val_losses, label='val_loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('Train and Validation Losses Across Steps')
    plt.legend()
    plt.savefig('train_val_loss')
    plt.show()
    

def main():
    plot_train_val_loss()

if __name__ == "__main__":
    main()