import torch
from config import device
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(model, data_loader, val_loader, model_path, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.BCELoss()
    model.train()
    best_val_loss = float("inf")
    for _ in range(epochs):
        model.train()
        epoch_loss = 0
        for data in data_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = loss_fn(out, data.y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data_loader)

        val_loss = val(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
        
        print(f"Epoch: {_}, Best: {best_val_loss}, Val Loss: {val_loss}")
    return best_val_loss


def val(model, data_loader):
    model.eval()
    loss = 0
    loss_fn = torch.nn.BCELoss()
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
        loss += loss_fn(out, data.y.float()).item()
    loss /= len(data_loader)
    return loss


def test(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_true.append(data.y)
        y_pred.append(
            out > 0.5
        )  # threshold because of sigmoid activation at last layer
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # TODO: add classification report

    return accuracy, precision, recall, f1
