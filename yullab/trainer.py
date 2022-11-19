import torch


def train(loader, model, criterion, optimizer, n_epochs):
    train_losses = []
    for epoch in range(n_epochs):
        running_loss = 0
        for images, labels in loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss/len(loader))
        print("Epoch: {}/{}.. ".format(epoch + 1, n_epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(loader)))

    return train_losses


def test(loader, model, criterion):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)

            test_loss += criterion(outputs, labels)

            top_p, top_class = outputs.topk(1, dim=1)
            _, label_idx = labels.topk(1, dim=1)

            equals = top_class == label_idx
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print("Test Loss: {:.3f}.. ".format(test_loss / len(loader)),
    "Test Accuracy: {:.3f}".format(accuracy / len(loader)))

    return test_loss, accuracy
