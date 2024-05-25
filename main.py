import torch.nn.functional
import customDataset
import utils
from torchvision.transforms import ToTensor, Compose, Normalize
from torch_snippets import *
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import PolynomialLR
import torchsummary
from tqdm import tqdm
from colorama import init, Fore, Style

init(convert=True)


class MultiHeadMGC(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(MultiHeadMGC, self).__init__()

        self.in_channels = in_channels

        self.chroma_head = self.cerberus_head()
        self.mel_spec_head = self.cerberus_head()
        self.mfcc_head = self.cerberus_head()

        self.drop = nn.Dropout(0.2)

        self.fc1 = nn.Linear(3 * 128 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, num_classes)

    def forward(self, x1, x2, x3):
        x1 = self.chroma_head(x1)
        x2 = self.mel_spec_head(x2)
        x3 = self.mfcc_head(x3)

        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        x3 = x3.reshape(x3.shape[0], -1)

        x = torch.cat((x1, x2, x3), dim=1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        x = self.fc5(x)

        return x

    def cerberus_head(self):
        head = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

        )

        return head


def eval_on_batch(input1, input2, input3, label, model, criterion, optimizer, train_on_batch=False):
    """
    Compute the loss on a batch of images
    :param input1: chromagram image
    :param input2: mel spectrogram image
    :param input3: mfcc image
    :param label: genre of the audio track
    :param model: model that is training
    :param criterion: loss function
    :param optimizer: method to update the gradient
    :param train_on_batch: flag to switch between evaluation on training/validation
    :return: loss on the batch
    """
    if train_on_batch:
        model.train()
        optimizer.zero_grad()
        output = model(input1, input2, input3)
        loss = criterion(output, label.squeeze(-1))
        loss.backward()
        optimizer.step()
    else:
        with torch.no_grad():
            model.eval()
            output = model(input1, input2, input3)
            loss = criterion(output, label.squeeze(-1))

    return loss


def check_accuracy(data_loader, model):
    """
    Computes the accuracy and the confusion matrix of the trained model on the test set
    :param data_loader: data on which the trained model is tested
    :param model: trained model
    """
    print()
    print(f'{Fore.LIGHTBLUE_EX}Checking accuracy on trained model{Style.RESET_ALL}')

    num_correct = 0
    num_samples = 0
    y_predictions = []
    y_true = []

    with torch.no_grad():
        model.eval()

        for x1, x2, x3, y in tqdm(data_loader):
            x1 = x1.to(device=device)
            x2 = x2.to(device=device)
            x3 = x3.to(device=device)

            y = y.to(device=device)

            scores = model(x1, x2, x3)
            _, predictions = scores.max(1)
            num_correct += (predictions == y.argmax()).sum()
            num_samples += predictions.size(0)

            y = y.data.cpu().argmax().numpy()
            y_true.append(y)
            predictions = predictions.data.cpu().numpy()
            y_predictions.extend(predictions)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}%')

        cf_matrix = confusion_matrix(y_true, y_predictions)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in utils.gtzan_genres],
                             columns=[i for i in utils.gtzan_genres])

        sn.heatmap(df_cm, annot=True)
        plt.show()


def save_state(epoch=None, model_dict=None, optimizer_dict=None, scheduler_dict=None, loss_dict=None,
               path='./mh_mgc.pth'):
    """
    Save not only the model, but also a series of info about the training
    :param epoch: epoch at which the data backup is executed
    :param model_dict: dictionary of the model
    :param optimizer_dict: dictionary of the optimizer
    :param scheduler_dict: dictionary of the learning rate scheduler
    :param loss_dict: loss function history
    :param path: path where to save all the information
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_dict,
        'optimizer_state_dict': optimizer_dict,
        'scheduler_state_dict': scheduler_dict,
        'loss_dict': loss_dict,

    }, path)


def training_loop(model, train_loader, val_loader, num_epochs, resume=False):
    """

    :param model: model to be trained
    :param train_loader: data for training
    :param val_loader: data for validation
    :param num_epochs: number of epochs
    :param resume: flag to allow uploading of the last model backup
    :return: trained model
    """
    learning_rate = 0.001

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if resume:
        print(f"{Fore.GREEN}Resume from checkpoint{Style.RESET_ALL}")
        print()
        model_dict = torch.load('./model.pth')

        start_epoch = model_dict['epoch']

        model.load_state_dict(model_dict['model_state_dict'])

        optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        scheduler = PolynomialLR(optimizer, power=2, total_iters=num_epochs)
        scheduler.load_state_dict(model_dict['scheduler_state_dict'])

        loss_dict = model_dict['loss_dict']

    else:
        start_epoch = 0
        scheduler = PolynomialLR(optimizer, power=2, total_iters=num_epochs)
        loss_dict = {
            'train': [],
            'val': [],
            'trigger': 0
        }

    last_train_loss = []
    last_val_loss = []

    for epoch in range(start_epoch, num_epochs):
        print(f'EPOCH: {epoch + 1}')
        print(f'Scheduler: Adjusting learning rate to {scheduler.get_last_lr()}')

        for ix, (data1, data2, data3, label) in enumerate(tqdm(train_loader, desc='Training')):
            loss = eval_on_batch(data1.to(device), data2.to(device), data3.to(device),
                                 label.to(device), model, criterion, optimizer, train_on_batch=True)
            last_train_loss.append(loss.detach().cpu().numpy())

        for ix, (data1, data2, data3, label) in enumerate(tqdm(val_loader, desc='Validation')):
            loss = eval_on_batch(data1.to(device), data2.to(device), data3.to(device),
                                 label.to(device), model, criterion, optimizer, train_on_batch=False)
            last_val_loss.append(loss.detach().cpu().numpy())

        avg_train_loss = np.mean(last_train_loss)
        avg_val_loss = np.mean(last_val_loss)

        print(f'[Avg train loss: {np.round(avg_train_loss, 4)} - Avg val loss: {np.round(avg_val_loss, 4)}]')
        print()

        loss_dict['train'].append(avg_train_loss)
        loss_dict['val'].append(avg_val_loss)

        if avg_train_loss < avg_val_loss:
            loss_dict['trigger'] += 1
            print(f"{Fore.RED}Early Stopping Trigger:{Style.RESET_ALL} {loss_dict['trigger']}")
            print()

            if loss_dict['trigger'] == 10:
                print(f"{Fore.RED}EXIT: Early Stopping in action!{Style.RESET_ALL}")
                save_state(epoch, model_dict=model.state_dict(), optimizer_dict=optimizer.state_dict(),
                           scheduler_dict=scheduler.state_dict(), loss_dict=loss_dict)
                break

        last_train_loss, last_val_loss = [], []

        if (epoch + 1) % 10 == 0:
            print(f'{Fore.GREEN}Saving model{Style.RESET_ALL}')
            print()
            save_state(epoch, model_dict=model.state_dict(), optimizer_dict=optimizer.state_dict(),
                       scheduler_dict=scheduler.state_dict(), loss_dict=loss_dict)

        scheduler.step()

    plt.plot(loss_dict['train'], color='C0', label='train loss')
    plt.plot(loss_dict['val'], color='C1', label='val_loss')
    plt.legend()
    plt.grid()
    plt.show()

    return model


def main():
    corrupt_data_path = './GTZAN/genres_original/jazz/jazz.00054.wav'
    if os.path.exists(corrupt_data_path):
        os.remove(corrupt_data_path)

    utils.create_images(create=False)

    images_chromas_list = []
    for filename in os.listdir('./GTZAN/images_3sec/chromas/'):
        images_chromas_list.append(filename)

    train, test = train_test_split(images_chromas_list, test_size=0.1)
    train, val = train_test_split(train, test_size=0.1)

    utils.create_annotations(create=True, lists=(train, val, test))

    img_transform = {
        'train': Compose([transforms.ToPILImage(), transforms.RandomResizedCrop(328),
                          transforms.RandomHorizontalFlip(), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])]),

        'val_test': Compose([transforms.ToPILImage(), transforms.Resize(360), transforms.CenterCrop(328), ToTensor(),
                             Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    }

    train_set, val_set, test_set = customDataset.get_datasets(transform=img_transform)

    batch_size = 64
    training_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mh_mgc = MultiHeadMGC().to(device=dev)
    print()
    print('Printing a summary of the model (wait just a sec)..')
    torchsummary.summary(mh_mgc, [(3, 328, 328), (3, 328, 328), (3, 328, 328)])

    trained_mh_mgc = training_loop(mh_mgc, train_loader=training_loader, val_loader=validation_loader, num_epochs=300,
                                   resume=False)

    check_accuracy(test_loader, trained_mh_mgc)


if __name__ == '__main__':
    main()
