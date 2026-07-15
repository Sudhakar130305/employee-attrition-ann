import matplotlib.pyplot as plt

def plot_loss(loss_history):

    plt.figure(figsize=(8,5))

    plt.plot(loss_history)

    plt.title("Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.show()

    import matplotlib.pyplot as plt


def plot_loss(
    train_loss,
    val_loss
):

    plt.figure(figsize=(10,6))

    plt.plot(
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid()

    plt.show()