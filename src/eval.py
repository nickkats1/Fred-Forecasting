import torch


def train_validate(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    optimizer,
    epochs,
    loss_fn
):
    """Training and validation loop for evaluating LSTM.
    
    Args:
        model: Type of RNN set to a device.
        X_train: training portion of data with features.
        y_train: training portion of data with targets.
        X_val: (1- train ratio) portion of data with features.
        y_val: (1-train_ratio) portion of data with targets.
        device: cuda if torch.cuda.is_available() else "cpu".
        optimizer: torch.optim.Adam with parameters of LSTM and learning rate.
        epochs: number of times model passes through data.
        loss_fn: MSELoss.
    
    Returns:
        Train Root Mean Squared Error (train RMSE) and validation Root Mean-Squared Error (val RMSE).
    """
    
    model = model.to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        train_loss = loss_fn(y_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_train = model(X_train)
            y_pred_val = model(X_val)

            train_rmse = torch.sqrt(loss_fn(y_pred_train, y_train)).item()
            val_rmse = torch.sqrt(loss_fn(y_pred_val, y_val)).item()

        print(f"Epoch {epoch+1}/{epochs} | train_rmse: {train_rmse:.4f} | val_rmse: {val_rmse:.4f}")
    