import matplotlib.pyplot as plt

def plot_loss(loss_history):

    plt.figure(figsize=(8,5))

    plt.plot(loss_history)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.show()