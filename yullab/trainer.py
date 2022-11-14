import torch


def train(loader, model, criterion, optimizer):
    running_loss = 0
    for imgs, labels in loader:
        optimizer.zero_grad()

        imgs = imgs.reshape(-1, 784)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        #train_losses.append(loss.item())

    #train_losses.append(running_loss/len(loader))
    return running_loss#, train_losses

def test(loader, model, criterion):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.reshape(-1, 784)
            outputs = model(images)

            test_loss += criterion(outputs, labels)

            top_p, top_class = outputs.topk(1, dim=1)
            _, label_idx = labels.topk(1, dim=1)

            equals = top_class == label_idx
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    #test_losses.append(test_loss/len(loader))
    return test_loss, accuracy #test_losses
