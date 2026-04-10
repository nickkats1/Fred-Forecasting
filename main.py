import torch
import torch.nn as nn
import torch.nn.functional as F


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)



from src.data.data_ingestion import fetch_data
from src.data.data_transformation import split_data, transform
from src.eval import train_validate
from src.models.lstm import LSTM
from src.utils import sliding_window, convert_array_to_tensor


# Pick series ID based on available fred data.
SERIES_ID = "DEXUSEU"


def main():
    
    
    # load in fred data with desired series id.
    df = fetch_data(SERIES_ID)
    
    # split data and choose portion to remain as training data.
    
    TRAIN_SIZE = 0.80
    
    df_train, df_val = split_data(df, TRAIN_SIZE)
    
    
    # scale both training and validation data.
    
    df_train_scaled, df_val_scaled = transform(df_train, df_val)
    
    
    # select a window size.
    
    WINDOW_SIZE = 20
    X_train, y_train = sliding_window(df_train_scaled, WINDOW_SIZE)
    X_val, y_val = sliding_window(df_val_scaled, WINDOW_SIZE)
    
    
    # use cuda if available, else, will train on CPU (cuda recommended)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # convert x_train, X_val, y_train, y_val to tensors.
    
    X_train = convert_array_to_tensor(X_train).to(device)
    y_train = convert_array_to_tensor(y_train).to(device)
    
    X_val = convert_array_to_tensor(X_val).to(device)
    y_val = convert_array_to_tensor(y_val).to(device)
    
    
    # set appropriate parameters for LSTM and set to device.
    
    model = LSTM(
    input_size=1,
    hidden_size=128,
    num_layers=2,
    output_size=1
    )
    model.to(device)
    
    
    # load in optimizer and loss function.
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    epochs = 100
    
    
    # train and validate model
    
    train_validate(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    optimizer,
    epochs,
    loss_fn
)
    
    # undue MinMaxScaler implemented from 'src.utils.transform'.
    
    scaler = MinMaxScaler()
    scaler.fit(df_train)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)


    y_pred_np = y_pred.detach().cpu().numpy()
    y_val_np = y_val.detach().cpu().numpy()

    pred_rescaled = scaler.inverse_transform(y_pred_np)
    actual_rescaled = scaler.inverse_transform(y_val_np)
    
    
    # metrics: r2_score, rmse, mae, mape
    
    r2 = r2_score(y_val_np, y_pred_np)
    print(f"R2 Score: {r2:.2f}")

    mape = mean_absolute_percentage_error(y_val_np, y_pred_np)
    print(f"Mean Absolute Percentage Error: {mape}")


    mae = mean_absolute_error(y_val_np, y_pred_np)
    print(f"Mean Absolute Error: {mae:.4f}")


    rmse = np.sqrt(mean_squared_error(y_val_np, y_pred_np))
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    
    # view predicted vs actual results
    
    train_len = int(len(df) * TRAIN_SIZE)
    val_target_dates = df["Date"].iloc[train_len + WINDOW_SIZE:].reset_index(drop=True)

    actual_flat = actual_rescaled.flatten()
    pred_flat = pred_rescaled.flatten()


    print("dates:", len(val_target_dates), "actual:", len(actual_flat), "pred:", len(pred_flat))

    comparison_df = pd.DataFrame({
        "Date": val_target_dates.values,
        "Actual": actual_flat,
        "Predicted": pred_flat
    })

    print(comparison_df.head(20))
    print(comparison_df.tail(20))
    
    

if __name__ == "__main__":
    main()
    
    

    
    
    





