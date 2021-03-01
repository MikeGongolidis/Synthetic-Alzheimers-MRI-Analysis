from torch import nn


class AryubiaNet(nn.Module):

    def __init__(self, num_classes: int = 4) -> None:
        super(AryubiaNet, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(256*3*3,eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(0.2),
            nn.Linear(256*3*3, 1280),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1280,eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(0.2),
            nn.Linear(1280, 1280),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1280,eps=1e-05, momentum=0.1, affine=True),
            nn.Dropout(0.4),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def train_model(model,train_loader,epoch,epochs,optimizer,current_lr,log_every):
    _ = model.train()

    y_preds = []
    y_trues = []
    losses = []
    
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        image.unsqueeze_(0)
        image = image.repeat(3,1,1,1)
        
        image = image.transpose(0,1)
        
        label = torch.autograd.Variable(label.long())
        prediction = model.forward(image.float())
        loss = criterion(prediction, label)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)


        if (i % log_every == 0) & (i > 0):
            print(
                '''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} |  lr : {5}'''.
                    format(
                    epoch + 1,
                    epochs,
                    i,
                    len(train_loader),
                    np.round(np.mean(losses), 4),
                    current_lr
                )
            )


    train_loss_epoch = np.round(np.mean(losses), 4)
    
    return train_loss_epoch


def evaluate_model(model, validation_loader, epoch, epochs, current_lr):
    _ = model.eval()


    y_trues = []
    y_preds = []
    losses = []

    for i, (image, label) in enumerate(validation_loader):
        
        image.unsqueeze_(0)
        image = image.repeat(3,1,1,1)

        image = image.transpose(0,1)

        label = torch.autograd.Variable(label.long())


        prediction = model.forward(image.float())

        loss = criterion(prediction, label)

        loss_value = loss.item()
        losses.append(loss_value)


        if (i % log_every == 0) & (i > 0):
            print(
                '''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} |  lr : {5}'''.
                    format(
                    epoch + 1,
                    epochs,
                    i,
                    len(validation_loader),
                    np.round(np.mean(losses), 4),
                    current_lr
                )
            )

    val_loss_epoch = np.round(np.mean(losses), 4)
    return val_loss_epoch  