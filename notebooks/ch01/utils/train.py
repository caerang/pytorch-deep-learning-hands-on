import torch
from tqdm import tqdm


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-"*80)

        for phase in ["train", "val"]:
            net.eval()
            if phase == "train":
                net.train()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == "train"):
                continue

            for inputs, labels in tqdm(dataloaders_dict[phase]):
                # optimizer 초기화
                optimizer.zero_grad()

                # feedforward
                with torch.set_grad_enabled(phase=="train"):
                    # 입력에 대해 예측 수행
                    outputs = net(inputs)
                    # 예측 결과와 GT 사이의 차이 계산 (손실값)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backpropagation if training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    # loss.item()이 반환하는 데이터 구조는 뭐지?
                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')